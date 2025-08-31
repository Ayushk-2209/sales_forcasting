from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

data = {
    'month': list(range(1, 11)), 
    'sales': [200, 220, 250, 270, 300, 320, 350, 370, 400, 420]
}
df = pd.DataFrame(data)

model = LinearRegression()
model.fit(df[['month']], df['sales'])

HTML_PAGE = """
<!doctype html>
<html>
    <head>
        <title>Sales Forecast with Graph</title>
    </head>
    <body>
        <h2>Sales Forecast Prediction</h2>
        <form method="get" action="/predict">
            <label>Enter Month:</label>
            <input type="number" name="month" required>
            <input type="submit" value="Predict">
        </form>

        {% if prediction is not none %}
            <h3>Predicted Sales for Month {{ month }}: {{ prediction }}</h3>
            <img src="data:image/png;base64,{{ plot_url }}" alt="Sales Chart"/>
        {% endif %}
    </body>
</html>
"""

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    month = None
    plot_url = None

    if request.method == 'POST':
        # API POST request
        month = request.json.get('month')
        if month is None or not isinstance(month, int):
            return jsonify({'error': 'Invalid input'}), 400
        prediction = round(float(model.predict([[month]])[0]), 2)
        return jsonify({'predicted_sales': prediction})

    else:
        month = request.args.get('month', type=int)
        if month is not None:
            prediction = round(float(model.predict([[month]])[0]), 2)

            # Generate graph
            all_months = df['month'].tolist() + [month]
            all_sales = df['sales'].tolist() + [prediction]

            plt.figure(figsize=(6,4))
            plt.plot(df['month'], df['sales'], marker='o', label='Historical Sales')
            plt.plot(month, prediction, marker='x', color='red', markersize=10, label='Predicted Sales')
            plt.xlabel('Month')
            plt.ylabel('Sales')
            plt.title('Sales Forecast')
            plt.legend()
            plt.tight_layout()

            # Save plot to PNG in memory
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template_string(HTML_PAGE, prediction=prediction, month=month, plot_url=plot_url)

@app.route('/')
def home():
    return render_template_string(HTML_PAGE, prediction=None, month=None, plot_url=None)

if __name__ == '__main__':
    app.run(debug=True)
