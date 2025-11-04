# PROJECT TITLE
Capstone Black Box functions

## NON-TECHNICAL EXPLANATION OF YOUR PROJECT
Project overview:
As part of the capstone project, we have been given eight synthetic black-box functions for optimising (BBO). Each of the function mimics real-world complexities, with feature such as non-linearity and multiple local maxima. We have been given an initial set of data and we are to query each function weekly and arrive at a query that gives the maximum value for each of the functions by the end of the project.
The BBO capstone project aims to simulate real world scenarios where, often, the data in the sparse, the optimum solution is unknown. The process to get the optimum solution requires domain knowledge, data-driven decision making, trial and error with different ML techniques and often intuition.
I work in the retail industry that collects a lot of data, both customer and transactional. Understanding customer behaviours and motivations is key to achieve growth in the business.

## DATA
Inputs and outputs:
There is an initial set of data, both input and output provided for each of the function. Using these I have developed a model that predicts the best next query.
The first week starts with initial set of inputs and outputs provided. The model that I have developed accepts these and input and generates the best query based on the values I have explained above.
I have plotted the inputs and outputs on a scatter plot and tried to identify the combinations of the inputs (features) that give the best query.

## CHALLENGES
Challenge objectives:
The eight BBO functions represent real life optimisation challenges. In some cases, the objective is a natural minimisation (e.g. side effects) in which case a transformation (such as negation) is applied. 
•	Technical approach
I have used Bayesian optimisation with GaussianProcessRegressor trained it with the following options.
•	Different acquisition strategies like PI, EI and UCB with different values 
•	Grid size of 100
•	RBF kernel.
Then I have chosen the query with the highest acquisition score.
## Weekly log
### Week 1: 
I have tried different acquisition functions, PI, EI, UCB and since the ask is to maximise, I have picked the points with the maximum value.
As the dimensions of the input increase, I have tried using different value for exploration parameter (kappa). It also tried using PI and EI with different values of the exploration bias (xi).
As I have worked through the functions, the input dimensions have increased and I have had to reduce the grid size to reduce computation requirements.
### Week 2:
In Week 1, I started getting errors when executing the code for determining the best query. I had to reduce the size of the grid to get the code to run.
This week I have kept the grid size constant (i.e. 100) but changed my code to reduce the number of points of computation by down sampling the total points by a picking point that are apart by a factor of the square of the dimension of the input. This has reduced the computational/memory requirements and still allows me to do exploration.
I considered logistic regression but it is most useful when the output a clear classification, ideally between two possibilities (Yes/No, On/Off etc…)
### Week 3:
I have stayed with Exploration this week because we still have a few more submission to go. As I get closer to the end, I will start exploiting. I have tried various values for xi (PI and EI) and kappa (UCB) and then selected the query with the best score.
I have increased the grid size to explore a larger area this has increased the number of computation but I have tweaked the down sampling to keep the computation requirements within reasonable limits.
In certain functions I have tried to use feature importance measure and RBF kernel to convert importance to length scales in the kernel which I passed to the GaussianProcessRegressor.
I considered SVM but that can be used only where there is a clear localisation of the optimum/maximised values. For example, function-1 where the radiation measurement is non-zero only in the region close to the contamination source. Similarly function-5 where which is unimodal with a single peak where yield is maximised. SVM may not be useful in for functions like function-3 (drug discovery) where the side-effects would change gradually or function 4 and function 6 where output will be improved gradually my fine tuning the hyper parameters. In essence the margins between the optimal value and others are very small.
