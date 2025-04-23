import os
import io
import csv
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/forecast', methods=['POST'])
def forecast():
    try:
        # Check if salary is provided
        if 'salary' not in request.form:
            return jsonify({'error': 'Monthly salary is required'}), 400
        
        try:
            salary = float(request.form['salary'])
            if salary <= 0:
                return jsonify({'error': 'Salary must be a positive number'}), 400
        except ValueError:
            return jsonify({'error': 'Salary must be a valid number'}), 400
        
        # Check if CSV file is provided
        if 'csv_file' not in request.files:
            return jsonify({'error': 'CSV file is required'}), 400
        
        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get forecast months
        try:
            forecast_months = int(request.form.get('forecast_months', 3))
            if forecast_months <= 0:
                return jsonify({'error': 'Forecast months must be a positive number'}), 400
        except ValueError:
            return jsonify({'error': 'Forecast months must be a valid number'}), 400
        
        # Read and validate CSV content
        try:
            csv_content = file.read().decode('utf-8')
            csv_file = io.StringIO(csv_content)
            csv_reader = csv.reader(csv_file)
            
            # Read header
            headers = next(csv_reader)
            if len(headers) != 3 or headers[0].lower() != 'month' or headers[1].lower() != 'category' or headers[2].lower() != 'amount':
                return jsonify({'error': 'CSV file must have headers: month,category,amount'}), 400
            
            # Parse data
            all_data = []
            months_found = set()
            
            for row in csv_reader:
                if len(row) != 3:
                    continue
                
                month_str, category, amount_str = row
                month_str = month_str.strip()
                category = category.strip()
                amount_str = amount_str.strip()
                
                if not category or not month_str:
                    continue
                
                try:
                    month = int(month_str)
                    amount = float(amount_str)
                except ValueError:
                    return jsonify({'error': f'Invalid month or amount for category {category}: {month_str}, {amount_str}'}), 400
                
                all_data.append({'category': category, 'amount': amount, 'month': month})
                months_found.add(month)
            
            if not all_data:
                return jsonify({'error': 'No valid data found in CSV file'}), 400
            
            # Check if we have at least 3 months of data
            if len(months_found) < 3:
                return jsonify({'error': f'At least 3 months of data required, but only {len(months_found)} months found'}), 400
            
            # Extract unique categories
            categories = list(set(item['category'] for item in all_data))
            
            # Generate forecast
            forecast_data = generate_forecast(all_data, categories, salary, forecast_months)
            
            return jsonify({
                'categories': categories,
                'forecast': forecast_data
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing CSV file: {str(e)}'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def generate_forecast(data, categories, salary, forecast_months):
    result = []
    
    # Get list of all months in the data
    months = sorted(list(set(item['month'] for item in data)))
    
    # Include existing months in the result (for charting)
    for month in months:
        month_data = {'month': month, 'savings': 0}
        total_expense = 0
        
        for category in categories:
            category_items = [item for item in data if item['category'] == category and item['month'] == month]
            if category_items:
                amount = sum(item['amount'] for item in category_items)
                month_data[category] = amount
                total_expense += amount
        
        month_data['savings'] = salary - total_expense
        result.append(month_data)
    
    # Calculate the next month number for forecasting
    next_month = max(months) + 1
    
    # Future months (forecast)
    for month in range(next_month, next_month + forecast_months):
        month_data = {'month': month, 'savings': 0}
        total_expense = 0
        
        for category in categories:
            # Get data for this category
            category_data = [item for item in data if item['category'] == category]
            
            if category_data:
                # Prepare data for linear regression
                # Group by month and sum amounts for each category-month combination
                monthly_category_sums = {}
                for item in category_data:
                    key = (item['month'], item['category'])
                    if key not in monthly_category_sums:
                        monthly_category_sums[key] = 0
                    monthly_category_sums[key] += item['amount']
                
                # Get X and y for regression
                X = []
                y = []
                for month_cat, amount in monthly_category_sums.items():
                    if month_cat[1] == category:
                        X.append([month_cat[0]])
                        y.append(amount)
                
                X = np.array(X)
                y = np.array(y)
                
                # Fit linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict amount for current month
                predicted_amount = max(0, float(model.predict([[month]])[0]))
                month_data[category] = predicted_amount
                total_expense += predicted_amount
        
        month_data['savings'] = salary - total_expense
        result.append(month_data)
    
    return result

if __name__ == '__main__':
    app.run(debug=True)