# Linear Regression Lab:

| Instructions: Makes sure 3 files are in the same folder: 1:  lin_reg.py  2: analysis.ipynb  3: housing.csv |
| ---------------------------------------------------------------------------------------------------------- |

This project uses a custom Python script and a Jupyter Notebook to predict house prices based on the data set provided. The main goal was to see which factors, like income or location, actually drive house prices and to show those results in a notebook file. I built it using **NumPy** for the math and **SciPy** for the statistical tests.

### The code:

The code uses NumPy to perform the matrix algebra required for the linear regression model. I used the normal equation to calculate the coefficients for each feature. I also used SciPy to handle the statistical distributions. Specifically, SciPy was used to calculate the p-values and critical values for the T-tests and F-test. Those are used by  the model to determine which features are statistically significant and to calculate the 95 percent confidence intervals(the confidence can be set to different values in the notebook). The library itself contains no print statements or comments to ensure the code remains focused purely on calculations.

### The Notebook:

The notebook was designed to meet the requirement of not using print statements or comments in the code cells. Instead of printing text, I used f-strings to combine labels and values into single objects. By placing these objects, lists, or dictionaries on the last line of a code cell, Jupyter displays them as a formatted output. This keeps the notebook clean while still showing the results clearly. I used markdown titles above each cell to explain what is happening in the code instead of using internal code comments.

I used the line np.set_printoptions(precision=2, suppress=True, linewidth=100) to improve the readability of certain outputs, especially the Pearsons' Correlation which returns a matrix that would be quite hard to read otherwise. 

### Data Handling and Cleaning:

I used **genfromtxt** to load the data instead of basic Python file handling because it is designed to process CSV files directly into NumPy arrays. It automatically handles things like headers and data types, which prevents manual errors when converting text into numbers for math, and saves a lot of time not having to do those things manually. 

To deal with missing values, I chose to remove those rows. While filling them with averages is an option, removing them ensures the model only learns from actual, recorded data. Since the dataset is large, losing a few rows doesn't hurt the model's performance and keeps the results more reliable. That's why I also included as an extra a small statistic showing how much data would be lost and which. 

For the categoric feature ( ocean proximity ), I transformed the categories into numbers using "one hot encoding". I excluded one category during this process to avoid a common mathematical issue where having too many related columns makes the model's results unstable (dummy variable trap). 

### Results Analysis:

With an R-squared of 0.65 we know our chosen features explain about 65% of the final price. 

The F-Test gave a p-value near zero which means the model is finding real patterns in the data, not just guessing.

T-Tests show that "median income" is the most significant contributor on the final price, meaning that in areas with higher incomes the house prices are also higher. Might also argue it's a circular cause-effect situation, might be that areas with higher income are more in demand or might be that due to gentrification or simply expensive areas attract mostly people who can afford that kind of price. 

On the other hand, the Pearson correlation matrix showed that features like total rooms and total bedrooms have a correlation score above 0.90. This tells us these two columns are almost identical in the information they provide, so if we were to analyze a limited number of features we might want to only choose the more generic one, in this case "total rooms" 

Finally, the categorical results for ocean proximity show a big difference in price based on location. The coefficients for being inland are much lower than the others, which means that location is a major driver in this specific dataset. The confidence intervals are narrow, which shows that our estimates are pretty precise. Since the range between the low and high values isn't very wide, it means the model found very consistent patterns across all the rows of data, making these results more reliable for this dataset.
