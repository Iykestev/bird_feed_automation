#!/usr/bin/env python3
"""
Real Bird Feed Data Analysis & Projection Model
Based on actual consumption data from handwritten records
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RealFeedDataAnalyzer:
    def __init__(self):
        """Initialize with actual feed consumption data from the images"""
        # Actual data from your handwritten records
        self.actual_data = {
            'Day': list(range(1, 43)),  # 42 days of data
            'Daily_Feed_Consumption': [
                15, 16, 20, 23, 26, 30, 35, 38, 42, 47, 52, 59, 62, 67, 73, 
                78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 139, 144, 148, 
                155, 159, 165, 170, 175, 179, 183, 188, 191, 196, 198, 203, 205, 208
            ],
            'Average_Weight': [
                56, 70, 87, 106, 129, 152, 179, 208, 241, 276, 315, 357, 402, 
                450, 501, 555, 612, 672, 734, 800, 868, 938, 1011, 1086, 1164, 
                1243, 1323, 1406, 1490, 1575, 1661, 1748, 1836, 1924, 2013, 
                2102, 2192, 2281, 2370, 2459, 2548, 2637
            ]
        }
        
        # Convert to DataFrame
        self.df = pd.DataFrame(self.actual_data)
        
        # Add calculated columns
        self.df['Date'] = pd.date_range(start='2024-01-01', periods=len(self.df))
        self.df['Week'] = ((self.df['Day'] - 1) // 7) + 1
        self.df['Feed_per_kg_bodyweight'] = self.df['Daily_Feed_Consumption'] / self.df['Average_Weight']
        
        # Calculate growth patterns
        self.df['Daily_Weight_Gain'] = self.df['Average_Weight'].diff()
        self.df['Feed_Efficiency'] = self.df['Daily_Weight_Gain'] / self.df['Daily_Feed_Consumption']
        
        print("✅ Real feed data loaded successfully!")
        print(f"📊 Data range: {len(self.df)} days")
        print(f"🐦 Current average weight: {self.df['Average_Weight'].iloc[-1]}g")
        print(f"🌾 Current daily feed: {self.df['Daily_Feed_Consumption'].iloc[-1]}kg")
    
    def analyze_consumption_patterns(self):
        """Analyze feed consumption patterns and trends"""
        analysis = {}
        
        # Basic statistics
        daily_feed = self.df['Daily_Feed_Consumption']
        analysis['daily_stats'] = {
            'mean': daily_feed.mean(),
            'median': daily_feed.median(),
            'std': daily_feed.std(),
            'min': daily_feed.min(),
            'max': daily_feed.max(),
            'trend_slope': stats.linregress(self.df['Day'], daily_feed).slope
        }
        
        # Weekly patterns
        weekly_consumption = self.df.groupby('Week')['Daily_Feed_Consumption'].agg(['sum', 'mean', 'std'])
        analysis['weekly_stats'] = weekly_consumption.to_dict()
        
        # Growth efficiency
        analysis['efficiency'] = {
            'avg_daily_gain': self.df['Daily_Weight_Gain'].mean(),
            'feed_conversion_ratio': self.df['Feed_Efficiency'].mean(),
            'feed_per_gram_gain': 1 / self.df['Feed_Efficiency'].mean()
        }
        
        # Consumption rate per body weight
        analysis['consumption_rate'] = {
            'avg_percentage_bodyweight': (self.df['Feed_per_kg_bodyweight'] * 1000).mean(),  # per kg to per g
            'current_rate': (self.df['Feed_per_kg_bodyweight'].iloc[-1] * 1000)
        }
        
        return analysis
    
    def predict_future_consumption(self, days_ahead=30):
        """Predict future consumption based on historical patterns"""
        # Fit polynomial trend to capture growth curve
        x = self.df['Day'].values
        y = self.df['Daily_Feed_Consumption'].values
        
        # Try different polynomial degrees and choose best fit
        best_r2 = 0
        best_degree = 1
        for degree in range(1, 6):
            coeffs = np.polyfit(x, y, degree)
            y_pred = np.polyval(coeffs, x)
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
            if r2 > best_r2:
                best_r2 = r2
                best_degree = degree
                best_coeffs = coeffs
        
        print(f"📈 Best fit: Polynomial degree {best_degree} (R² = {best_r2:.3f})")
        
        # Generate future predictions
        future_days = np.arange(len(self.df) + 1, len(self.df) + days_ahead + 1)
        future_consumption = np.polyval(best_coeffs, future_days)
        
        # Create future dates
        last_date = self.df['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead)
        
        # Create prediction DataFrame
        prediction_df = pd.DataFrame({
            'Day': future_days,
            'Date': future_dates,
            'Predicted_Daily_Feed_kg': np.maximum(future_consumption, 0),  # Ensure positive values
            'Week': ((future_days - 1) // 7) + 1,
            'Confidence_Lower': future_consumption * 0.9,  # Simple confidence interval
            'Confidence_Upper': future_consumption * 1.1
        })
        
        return prediction_df, best_coeffs, best_r2
    
    def calculate_feed_bags_needed(self, feed_kg_per_day, bag_size_kg=25):
        """Convert feed requirements to number of bags needed"""
        daily_bags = feed_kg_per_day / bag_size_kg
        return {
            'daily_bags': daily_bags,
            'daily_bags_rounded': np.ceil(daily_bags),
            'weekly_bags': daily_bags * 7,
            'monthly_bags': daily_bags * 30
        }
    
    def create_comprehensive_report(self, projection_days=30):
        """Create comprehensive analysis and projection report"""
        analysis = self.analyze_consumption_patterns()
        predictions, coeffs, r2 = self.predict_future_consumption(projection_days)
        
        # Combine historical and predicted data
        historical_df = self.df[['Day', 'Date', 'Daily_Feed_Consumption']].copy()
        historical_df['Data_Type'] = 'Historical'
        historical_df.rename(columns={'Daily_Feed_Consumption': 'Feed_kg'}, inplace=True)
        
        prediction_df = predictions[['Day', 'Date', 'Predicted_Daily_Feed_kg']].copy()
        prediction_df['Data_Type'] = 'Predicted'
        prediction_df.rename(columns={'Predicted_Daily_Feed_kg': 'Feed_kg'}, inplace=True)
        
        combined_df = pd.concat([historical_df, prediction_df], ignore_index=True)
        
        # Add bag calculations
        combined_df['Daily_Bags_25kg'] = combined_df['Feed_kg'] / 25
        combined_df['Daily_Bags_Rounded'] = np.ceil(combined_df['Daily_Bags_25kg'])
        combined_df['Cumulative_Feed_kg'] = combined_df['Feed_kg'].cumsum()
        combined_df['Cumulative_Bags'] = combined_df['Daily_Bags_25kg'].cumsum()
        
        return combined_df, analysis, predictions, r2
    
    def export_to_excel(self, filename="real_bird_feed_analysis.xlsx"):
        """Export comprehensive analysis to Excel"""
        try:
            combined_df, analysis, predictions, r2 = self.create_comprehensive_report()
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Main data with predictions
                combined_df.to_excel(writer, sheet_name='Complete_Data', index=False)
                
                # Historical data only
                self.df.to_excel(writer, sheet_name='Historical_Data', index=False)
                
                # Future predictions only
                predictions.to_excel(writer, sheet_name='Future_Predictions', index=False)
                
                # Analysis summary
                analysis_rows = []
                
                # Daily statistics
                for key, value in analysis['daily_stats'].items():
                    analysis_rows.append(['Daily Statistics', key.replace('_', ' ').title(), f"{value:.2f}"])
                
                # Efficiency metrics
                for key, value in analysis['efficiency'].items():
                    analysis_rows.append(['Efficiency', key.replace('_', ' ').title(), f"{value:.3f}"])
                
                # Consumption rate
                for key, value in analysis['consumption_rate'].items():
                    analysis_rows.append(['Consumption Rate', key.replace('_', ' ').title(), f"{value:.2f}"])
                
                analysis_df = pd.DataFrame(analysis_rows, columns=['Category', 'Metric', 'Value'])
                analysis_df.to_excel(writer, sheet_name='Analysis_Summary', index=False)
                
                # Weekly summary
                weekly_data = []
                for week in range(1, int(combined_df['Day'].max() // 7) + 2):
                    week_data = combined_df[combined_df['Day'].between((week-1)*7+1, week*7)]
                    if not week_data.empty:
                        weekly_data.append([
                            week,
                            week_data['Date'].min().strftime('%Y-%m-%d'),
                            week_data['Date'].max().strftime('%Y-%m-%d'),
                            week_data['Feed_kg'].sum(),
                            week_data['Daily_Bags_25kg'].sum(),
                            np.ceil(week_data['Daily_Bags_25kg'].sum()),
                            week_data['Data_Type'].iloc[0]
                        ])
                
                weekly_df = pd.DataFrame(weekly_data, columns=[
                    'Week', 'Start_Date', 'End_Date', 'Total_Feed_kg', 
                    'Total_Bags', 'Bags_Rounded', 'Data_Type'
                ])
                weekly_df.to_excel(writer, sheet_name='Weekly_Summary', index=False)
                
                # Parameters and metadata
                params_data = [
                    ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    ['Historical Data Days', len(self.df)],
                    ['Prediction Days', len(predictions)],
                    ['Model Accuracy (R²)', f"{r2:.3f}"],
                    ['Current Daily Feed (kg)', f"{self.df['Daily_Feed_Consumption'].iloc[-1]}"],
                    ['Current Avg Weight (g)', f"{self.df['Average_Weight'].iloc[-1]}"],
                    ['Bag Size (kg)', '25'],
                    ['Current Daily Bags Needed', f"{self.df['Daily_Feed_Consumption'].iloc[-1] / 25:.2f}"]
                ]
                
                params_df = pd.DataFrame(params_data, columns=['Parameter', 'Value'])
                params_df.to_excel(writer, sheet_name='Parameters', index=False)
            
            print(f"✅ Excel export completed: {filename}")
            print(f"📊 Contains {len(combined_df)} total data points")
            print(f"📈 Prediction accuracy: {r2:.1%}")
            
            return filename
            
        except Exception as e:
            print(f"❌ Export error: {e}")
            return None
    
    def plot_analysis(self):
        """Create comprehensive visualization plots"""
        combined_df, analysis, predictions, r2 = self.create_comprehensive_report()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Bird Feed Consumption Analysis & Projections', fontsize=16, fontweight='bold')
        
        # Plot 1: Historical vs Predicted Consumption
        historical = combined_df[combined_df['Data_Type'] == 'Historical']
        predicted = combined_df[combined_df['Data_Type'] == 'Predicted']
        
        axes[0, 0].plot(historical['Day'], historical['Feed_kg'], 'b-', label='Historical', linewidth=2)
        axes[0, 0].plot(predicted['Day'], predicted['Feed_kg'], 'r--', label='Predicted', linewidth=2)
        axes[0, 0].set_title(f'Daily Feed Consumption (R² = {r2:.3f})')
        axes[0, 0].set_xlabel('Day')
        axes[0, 0].set_ylabel('Feed (kg)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Feed Consumption
        axes[0, 1].plot(combined_df['Day'], combined_df['Cumulative_Feed_kg'], 'g-', linewidth=2)
        axes[0, 1].axvline(x=len(self.df), color='red', linestyle=':', label='Prediction Start')
        axes[0, 1].set_title('Cumulative Feed Consumption')
        axes[0, 1].set_xlabel('Day')
        axes[0, 1].set_ylabel('Total Feed (kg)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Daily Bags Required
        axes[0, 2].plot(combined_df['Day'], combined_df['Daily_Bags_25kg'], 'purple', linewidth=2)
        axes[0, 2].axvline(x=len(self.df), color='red', linestyle=':', alpha=0.7)
        axes[0, 2].set_title('Daily Bags Required (25kg bags)')
        axes[0, 2].set_xlabel('Day')
        axes[0, 2].set_ylabel('Bags per Day')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Feed Efficiency Over Time
        axes[1, 0].plot(self.df['Day'], self.df['Feed_Efficiency'], 'orange', marker='o', markersize=3)
        axes[1, 0].set_title('Feed Conversion Efficiency')
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Weight Gain per kg Feed')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Weekly Consumption Pattern
        weekly_consumption = combined_df.groupby(((combined_df['Day'] - 1) // 7) + 1)['Feed_kg'].sum()
        axes[1, 1].bar(weekly_consumption.index, weekly_consumption.values, 
                      color=['blue' if x <= len(self.df)//7 else 'red' for x in weekly_consumption.index],
                      alpha=0.7)
        axes[1, 1].set_title('Weekly Feed Consumption')
        axes[1, 1].set_xlabel('Week')
        axes[1, 1].set_ylabel('Total Feed (kg)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Growth vs Feed Relationship
        axes[1, 2].scatter(self.df['Average_Weight'], self.df['Daily_Feed_Consumption'], 
                          c=self.df['Day'], cmap='viridis', s=50)
        axes[1, 2].set_title('Feed vs Average Weight')
        axes[1, 2].set_xlabel('Average Weight (g)')
        axes[1, 2].set_ylabel('Daily Feed (kg)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print key insights
        print("\n" + "="*60)
        print("📊 KEY INSIGHTS FROM YOUR DATA:")
        print("="*60)
        print(f"🐦 Birds are gaining {analysis['efficiency']['avg_daily_gain']:.1f}g per day on average")
        print(f"🌾 Current daily feed requirement: {self.df['Daily_Feed_Consumption'].iloc[-1]}kg")
        print(f"📦 Current daily bags needed: {self.df['Daily_Feed_Consumption'].iloc[-1]/25:.1f} bags")
        print(f"📈 Feed consumption is increasing by {analysis['daily_stats']['trend_slope']:.2f}kg per day")
        print(f"💰 At $50/bag, daily feed cost: ${(self.df['Daily_Feed_Consumption'].iloc[-1]/25)*50:.2f}")

def main():
    """Main function to run the analysis"""
    print("🐦 REAL BIRD FEED DATA ANALYSIS")
    print("="*50)
    
    # Initialize analyzer with real data
    analyzer = RealFeedDataAnalyzer()
    
    # Run comprehensive analysis
    print("\n📊 Analyzing consumption patterns...")
    analysis = analyzer.analyze_consumption_patterns()
    
    # Generate predictions
    print("📈 Generating future projections...")
    predictions, coeffs, r2 = analyzer.predict_future_consumption(30)
    
    # Create visualizations
    print("📊 Creating analysis charts...")
    analyzer.plot_analysis()
    
    # Export to Excel
    print("📁 Exporting to Excel...")
    excel_file = analyzer.export_to_excel()
    
    print(f"\n🎉 Analysis complete! Check {excel_file} for detailed results.")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()