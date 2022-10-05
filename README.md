First things first, we need some data. We are using data from one of the most esteemed research colleges in the world to predict the weather using machine learning. 
We will presume that the data in the dataset is accurate.
 
1. Data Preparation
Unfortunately, we're still not at the point where we can simply feed a model the raw data and have it respond. To incorporate our data into a machine learning model,
we will need to make a few simple adjustments. A certain amount of data manipulation will be necessary, but the precise procedures in data preparation will depend 
on the model employed and the data obtained. We have created a function called wrangle () and called it with the name of our dataframe. To prevent the original 
dataframe from being damaged, we want to create a clone of it. After that, we'll get rid of the columns with a lot of cardinality.

Columns with a high cardinality are those whose values are extremely uncommon or singular. Given the prevalence of high cardinality data in most time-series datasets,
we will immediately address this issue by eliminating all high cardinality columns from our dataset in order to prevent future confusion in our model.

2. By developing a method, we turned temperature and the columns into DateTime objects. after invoking our wrangle function on it, a cleaned-up version of our global 
temp dataframe with no missing values.

3. Visualization
Before moving further with the development of a machine learning-based weather prediction model, we displayed this data to identify correlations between the data.
The columns that we have continued using have a strong correlation with one another.

4. Separating Our Target to Predict Weather. 
Now divide the data into targets and features. In this scenario, the actual average land and ocean temperature and characteristics are all the columns the model 
utilizes to generate a prediction. The target, often known as Y, is the value we want to predict.

5. Train Test Split
Now, using scikit-train test split learn's technique, we divided the data in order to build a machine learning model for weather prediction.

6. Baseline Mean Absolute Error
We must first set a baseline, a reasonable metric that we want to outperform using our model, before we can make and assess any predictions using our machine learning
model to forecast the weather. We should try an alternative model or acknowledge that machine learning is not a good fit for our situation if our model cannot 
improve from the starting point.

7.Now using machine learning to predict the weather We will develop a Random Forest algorithm that can carry out both the classification and regression tasks. 
We can determine the precision of our forecasts by subtracting the average percentage error from 100%. With machine learning, our model has learned to predict 
the weather for the upcoming year with 99% accuracy.
