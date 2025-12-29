# PROJECT TITLE
Capstone Black Box functions

## NON-TECHNICAL EXPLANATION OF THE PROJECT
Project overview:
As part of the capstone project, we have been given eight synthetic black-box functions for optimising (BBO). Each of the function mimics real-world complexities, with feature such as non-linearity and multiple local maxima. We have been given an initial set of data and we are to query each function weekly and arrive at a query that gives the maximum value for each of the functions by the end of the project.
The BBO capstone project aims to simulate real world scenarios where, often, the data is sparse, the optimum solution is unknown. The process to get the optimum solution requires domain knowledge, data-driven decision making, trial and error with different ML techniques and often intuition.

## DATA
Inputs and outputs:
There is an initial set of data, both input and output provided for each of the function. Using these I have developed a model that predicts the best next query.
The first week starts with initial set of inputs and outputs provided. The model that I have developed accepts these and input and generates the best query based on the values I have explained above.
I have plotted the inputs and outputs on a scatter plot and tried to identify the combinations of the inputs (features) that give the best query.
A detailed description is available in the project’s [Data Sheet](/docs/Datasheet%20for%20BBO%20Capstone%20project.pdf).

## CHALLENGES
Challenge objectives:
The eight BBO functions represent real life optimisation challenges. In some cases, the objective is a natural minimisation (e.g. side effects) in which case a transformation (such as negation) is applied. 
A detailed description is available in the [Model Card](/docs/Model%20Card%20BBO%20Capstone%20Project.pdf)

This [series of plots](/results/Results%20by%20function.md) show the results of the model for each of the functions over the past few weeks. 

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
### Week 4:
I have established a flow for evaluating the grid and identifying the candidate points/queries for submission.
This week I have focused on reducing computational requirement by filtering the grid using SVM. Based on the requirement and behaviour of functions I have used, Gaussian Process, Median or K nearest neighbour techniques. I have retained my explorattive stance. Filtering has allowed me to keep the exploration size large and yet reduce the number of candidate points to evaluate. This is in addition to the downsampling that I introduced in Week 2.
### Week 5:
I have developed two neural networks. One that is Fixed and the other that uses gradient descent and backpropagation. Compared the results of the two. Then, to reduce the number of epochs I have introduced a early stopping, the iterations stop once the loss falls below a certain threshold. I have used another value called patience that allows the iterations to continue for a fixed number of times even after the loss threshold is reached.
The steps i have introduced are 
1) Read input data
2) Add the results of the previously submitted queries.
3) Train the surrogate on the input
4) Create/Prepare the grid, downsample, filter using SVM (gp, median, knn).
5) Predict the output for the filtered grid using the trained surrogate.
6) Apply Beysian optimisation/compute acqusition for various values of XI, PI, Kappa.
7) Find best candidate.
### Week 6:
I have made several changes to my code base. I have moved towards using a configuration file. The config file has my hyperparameters for model training like learning rate, epochs, patience etc... and hyperparameters for optimisation like xi, kappa, grid size, filter strategy etc... Then there are two config variables optimization_direction and objective_mode.
Optimisation direction has the following values
• For "min", switched to minimization PI/EI formulas (using y_min instead of y_max).
• For "min", UCB becomes a Lower Confidence Bound analogue (-mean + kappa*std).
And objective_mode has the following values.
• Raw: Train on the actual outputs, optimise them directly.
• Zero_target: Train on squared outputs, optimise distance to zero.
• Negated: Flip the sign so that maximisation logic corresponds to minimisation.

Another general change that I have made is to the create_bound function to such that every grid or candidate generated from these bounds will respect the constraint that individual inputs cannot be negative.

To keep record of the various runs of the model I am now recording the output candidate points and plots in the results folder with the timestamp of the run in the folder name.
### Week 7:
For higher dimension functions the grid size becomes too large so I have tried using Latin Hypercude Sampling and Sobol in addition to Cartesian (the simple grid size) this week to find candidate points/queries.
I have done this introducing a config parameter called 'sample_strategy'
    sample_strategy: "lhs"   # cartesian / latin hypercube sampling (lhs) / sobol
I have renamed the file containing my submissions and their outputs more sensibly.
### Week 8:
I have used the emergence features of the OpenAI GPT-4.1 model to get candidate points/queries this week.
I have added cofiguration that allows me to switch between the LLM and other models I have developed so far.
### Week 9:
This week I have gone a step further in the details I have provided in the prompts. I have included specific characteristics of the function for which I am asking the candidate point. For example, for function 1, I have added the following text to the prompt
"Note that only proximity yields a non-zero reading in a contamination radiation field"
### Week 10:
Moved away from using llm prompts and back to using the surrogate model and bayesian optimisation techniques.
Function 1: Continued to use SVM filter because there is a clear boundary between high value and low value areas.
Added a plot to show the SVM filtering in action.
Function 2: Removed the use of SVM filter because 
• The function is not sparse
• There is no zero‑region
• There is no classification boundary
• Every point returns a meaningful value
• Filtering would remove useful exploration regions
Function 3: Removed the use of GP filter since it might over smoothen. The surrogate already smoothens the noise.
Function 4: The function is in 4D space and it has several local optima, so I have made the grid size really big and used random and sobol sampling to reduce the computation required. The attempt has been to cast the net as wide as possible.
Function 5: This function is a unimodal function (single peak) therefore in this weeks submissions, i have focused on exploitation and fast convergence to the peak.
Function 6: The function is unimodal function so for week 10 I have shifted focus to exploitation and quick convergence. Sampling on 3000 points using sobol.
Function 7: Balanced exploration with exploitation to get the point with the best score. Sticking with sobol as the sampling strategy because of high dimensions. Sampling 4000 points.
Function 8: Quite similar to function 6 and 4, I have balanced between exploring and exploitation (EI and UCB). Stuck to sobol with 10000 sample points.
