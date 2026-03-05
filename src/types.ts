export interface Dataset {
  id: string;
  name: string;
  description: string;
  type: 'classification' | 'regression' | 'clustering';
  features: string[];
  target?: string;
  columnDescriptions?: Record<string, string>;
  data: any[];
}

export type AlgorithmType = 'Regression' | 'Classification' | 'Clustering';

export interface Hyperparameter {
  id: string;
  name: string;
  min: number;
  max: number;
  step: number;
  default: number;
  description: string;
}

export interface AlgorithmInfo {
  id: string;
  name: string;
  type: AlgorithmType;
  description: string;
  whenToUse: string;
  howItWorks: string[];
  mathIntuition?: string;
  realWorldCase: string;
  hyperparameters?: Hyperparameter[];
  complexityParameter?: string;
  educationalDetails?: {
    whatItDoes: string;
    howItWorks: string;
    mathIntuition: string;
    realWorldExamples: string[];
  };
}

export interface HistoryItem {
  id: number;
  algo_id: string;
  algo_name: string;
  dataset_id: string;
  dataset_name: string;
  features: string[];
  target: string;
  hyperparams: Record<string, any>;
  metrics: Record<string, any>;
  created_at: string;
}

export const ALGORITHMS: AlgorithmInfo[] = [
  // Regression
  {
    id: 'linear-regression',
    name: 'Linear Regression',
    type: 'Regression',
    description: 'Models the relationship between a dependent variable and one or more independent variables by fitting a linear equation.',
    whenToUse: 'When you want to predict a continuous value and assume a linear relationship.',
    howItWorks: [
      'Finds the best-fitting straight line through the data points.',
      'Minimizes the sum of squared differences between observed and predicted values.',
      'Assumes a constant rate of change between variables.'
    ],
    mathIntuition: 'y = mx + b (where m is the slope and b is the intercept)',
    realWorldCase: 'Predicting house prices based on square footage.',
    educationalDetails: {
      whatItDoes: 'Linear Regression is the simplest form of regression. It attempts to model the relationship between two variables by fitting a linear equation to observed data. One variable is considered to be an explanatory variable, and the other is considered to be a dependent variable.',
      howItWorks: 'The algorithm uses the "Least Squares" method to find the best-fitting line. It calculates the distance from each data point to the line, squares these distances (to remove negative values), and then minimizes the sum of these squares. This ensures the line is as close as possible to all points simultaneously.',
      mathIntuition: 'Think of it as finding the "average" trend in a scatter plot. The slope (m) tells you how much the output changes for every unit of input, and the intercept (b) is the starting value when input is zero.',
      realWorldExamples: [
        'Predicting sales revenue based on advertising spend.',
        'Estimating the impact of temperature on electricity consumption.',
        'Forecasting crop yields based on rainfall amounts.'
      ]
    }
  },
  {
    id: 'multiple-linear-regression',
    name: 'Multiple Linear Regression',
    type: 'Regression',
    description: 'An extension of linear regression that uses multiple explanatory variables to predict the outcome of a response variable.',
    whenToUse: 'When you have multiple features influencing a single continuous target.',
    howItWorks: [
      'Extends linear regression to multiple dimensions.',
      'Assigns a weight (coefficient) to each input feature.',
      'Combines all features linearly to make a final prediction.'
    ],
    mathIntuition: 'y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ',
    realWorldCase: 'Predicting a student\'s GPA based on study hours, attendance, and previous grades.',
    educationalDetails: {
      whatItDoes: 'Multiple Linear Regression (MLR) is used to explain the relationship between one continuous dependent variable and two or more independent variables. It helps in understanding how much of the variation in the output can be explained by the input features together.',
      howItWorks: 'Similar to simple linear regression, it uses Ordinary Least Squares (OLS) to estimate coefficients. Each coefficient represents the change in the output for a one-unit change in that specific input, assuming all other inputs stay constant.',
      mathIntuition: 'Imagine a flat plane (in 3D) or a "hyperplane" (in higher dimensions) that slices through the data points. The goal is to orient this plane so that it minimizes the total error across all dimensions.',
      realWorldExamples: [
        'Predicting car fuel efficiency (MPG) based on weight, horsepower, and engine size.',
        'Estimating the value of a company based on its assets, liabilities, and quarterly growth.',
        'Determining the risk of heart disease based on age, BMI, and blood pressure.'
      ]
    }
  },
  {
    id: 'polynomial-regression',
    name: 'Polynomial Regression',
    type: 'Regression',
    description: 'A form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modelled as an nth degree polynomial.',
    whenToUse: 'When the data shows a non-linear, curved relationship.',
    howItWorks: [
      'Fits a curve rather than a straight line.',
      'Transforms features by raising them to a power (e.g., x²).',
      'Captures non-linear patterns in the data.'
    ],
    mathIntuition: 'y = β₀ + β₁x + β₂x² + ... + βₙxⁿ',
    realWorldCase: 'Modeling the growth rate of a bacterial population over time.',
    hyperparameters: [
      { id: 'degree', name: 'Degree', min: 1, max: 5, step: 1, default: 2, description: 'The power of the polynomial (higher = more complex curve).' }
    ],
    complexityParameter: 'degree',
    educationalDetails: {
      whatItDoes: 'Polynomial Regression is a special case of linear regression where we add powers of the original features as new features. This allows the model to fit curves and more complex shapes that a straight line cannot capture.',
      howItWorks: 'It transforms the input data by creating new columns for x², x³, etc. Then, it applies standard linear regression to these transformed features. Even though the relationship between x and y is non-linear, the relationship between the *transformed* features and y is still linear.',
      mathIntuition: 'Think of it as "bending" the regression line. A degree-2 polynomial creates a parabola (U-shape), while higher degrees allow for more "wiggles" in the curve to fit the data points more closely.',
      realWorldExamples: [
        'Modeling the trajectory of a projectile under gravity.',
        'Predicting the yield of a chemical reaction based on temperature.',
        'Analyzing the relationship between income and happiness (which often plateaus).'
      ]
    }
  },
  // Classification
  {
    id: 'logistic-regression',
    name: 'Logistic Regression',
    type: 'Classification',
    description: 'Despite its name, it is a classification algorithm used to estimate the probability of a binary response.',
    whenToUse: 'When you need to classify data into two distinct categories (e.g., Yes/No).',
    howItWorks: [
      'Uses a sigmoid function to map any value to a probability between 0 and 1.',
      'Sets a threshold (usually 0.5) to decide the final class.',
      'Finds the best boundary that separates the two classes.'
    ],
    mathIntuition: 'P(y=1) = 1 / (1 + e⁻ᶻ), where z is a linear combination of features.',
    realWorldCase: 'Predicting whether an email is spam or not.',
    hyperparameters: [
      { id: 'learningRate', name: 'Learning Rate', min: 0.001, max: 0.1, step: 0.001, default: 0.01, description: 'Controls how much the model changes in response to the estimated error each time the model weights are updated.' },
      { id: 'numSteps', name: 'Iterations', min: 100, max: 5000, step: 100, default: 1000, description: 'The number of times the algorithm goes through the training data.' }
    ],
    educationalDetails: {
      whatItDoes: 'Logistic Regression is used for binary classification. It predicts the probability that a given input belongs to a specific class (e.g., "Spam" or "Not Spam"). It is the foundation for many neural network architectures.',
      howItWorks: 'It calculates a weighted sum of the inputs (like linear regression) and then passes that sum through a "Sigmoid" function. This function squashes any input value into a range between 0 and 1, which we interpret as a probability.',
      mathIntuition: 'Instead of a straight line, it fits an "S-shaped" curve. When the input is very high, the probability is near 1; when it is very low, the probability is near 0. The middle part of the S is where the decision is most uncertain.',
      realWorldExamples: [
        'Predicting if a credit card transaction is fraudulent.',
        'Determining if a patient has a specific disease based on symptoms.',
        'Forecasting if a customer will "churn" (cancel their subscription).'
      ]
    }
  },
  {
    id: 'knn',
    name: 'K-Nearest Neighbors (KNN)',
    type: 'Classification',
    description: 'A non-parametric method that classifies a data point based on how its neighbors are classified.',
    whenToUse: 'When you have a small dataset with clear boundaries between classes.',
    howItWorks: [
      'Stores all available cases and classifies new cases based on a similarity measure.',
      'Finds the "K" closest data points in the training set.',
      'Assigns the most common class among those neighbors to the new point.'
    ],
    mathIntuition: 'Distance = √Σ(xᵢ - yᵢ)² (Euclidean Distance)',
    realWorldCase: 'Recommending similar products to a customer based on their purchase history.',
    hyperparameters: [
      { id: 'k', name: 'K (Neighbors)', min: 1, max: 21, step: 2, default: 3, description: 'The number of nearest neighbors to consider.' }
    ],
    complexityParameter: 'k',
    educationalDetails: {
      whatItDoes: 'KNN is a "lazy learner" because it doesn\'t actually "learn" a model during training. Instead, it memorizes the training data and does all the work when you ask for a prediction.',
      howItWorks: 'When a new point arrives, KNN looks at the "K" points in the training set that are physically closest to it. It then takes a "majority vote"—if 4 out of 5 neighbors are "Blue", the new point is classified as "Blue".',
      mathIntuition: 'Birds of a feather flock together. Points that are close to each other in the feature space are likely to belong to the same category. The choice of K is critical: too small and it\'s sensitive to noise; too large and it blurs the boundaries.',
      realWorldExamples: [
        'Handwriting recognition (comparing a new letter to known samples).',
        'Recommender systems (finding users with similar movie tastes).',
        'Detecting anomalies in network traffic based on typical patterns.'
      ]
    }
  },
  {
    id: 'decision-tree',
    name: 'Decision Tree',
    type: 'Classification',
    description: 'A tree-like model of decisions and their possible consequences, including chance event outcomes.',
    whenToUse: 'When you need an interpretable model that handles both numerical and categorical data.',
    howItWorks: [
      'Splits the data into subsets based on the most significant feature.',
      'Repeats the process recursively to create branches.',
      'The "leaves" of the tree represent the final classification.'
    ],
    mathIntuition: 'Uses Gini Impurity or Information Gain to decide where to split.',
    realWorldCase: 'A bank deciding whether to approve a loan based on credit score, income, and age.',
    hyperparameters: [
      { id: 'maxDepth', name: 'Max Depth', min: 1, max: 20, step: 1, default: 5, description: 'The maximum number of levels in the tree.' }
    ],
    complexityParameter: 'maxDepth',
    educationalDetails: {
      whatItDoes: 'Decision Trees break down a complex decision-making process into a series of simple, binary questions (e.g., "Is income > $50k?"). This makes them extremely easy for humans to understand and visualize.',
      howItWorks: 'At each step, the algorithm looks for the feature and the specific value that "purifies" the data the most. It wants to split the data so that the resulting groups are as homogeneous as possible (e.g., one group is mostly "Yes" and the other is mostly "No").',
      mathIntuition: 'It\'s like playing a game of 20 Questions. The goal is to ask the most informative questions first to narrow down the possibilities as quickly as possible.',
      realWorldExamples: [
        'Medical diagnosis flowcharts.',
        'Customer support chatbots routing users to the right department.',
        'Predicting whether a flight will be delayed based on weather and airline.'
      ]
    }
  },
  {
    id: 'random-forest',
    name: 'Random Forest',
    type: 'Classification',
    description: 'An ensemble learning method that operates by constructing a multitude of decision trees at training time.',
    whenToUse: 'When you need high accuracy and want to avoid overfitting common in single decision trees.',
    howItWorks: [
      'Creates many decision trees on random subsets of the data.',
      'Each tree gives a "vote" for the classification.',
      'The class with the most votes becomes the final prediction.'
    ],
    mathIntuition: 'Wisdom of the crowd: aggregating many weak models into one strong model.',
    realWorldCase: 'Predicting stock market trends by combining multiple indicators.',
    hyperparameters: [
      { id: 'nEstimators', name: 'Estimators', min: 5, max: 100, step: 5, default: 20, description: 'The number of trees in the forest.' },
      { id: 'maxDepth', name: 'Max Depth', min: 1, max: 20, step: 1, default: 5, description: 'The maximum depth of each tree.' }
    ],
    complexityParameter: 'nEstimators',
    educationalDetails: {
      whatItDoes: 'Random Forest is an "Ensemble" method. Instead of relying on one "expert" (a single tree), it consults a "crowd" of trees. This makes it much more robust and less likely to be fooled by noise in the data.',
      howItWorks: 'It uses two types of randomness: 1) It trains each tree on a random subset of the data (Bagging), and 2) It only considers a random subset of features for each split. This ensures the trees are diverse and don\'t all make the same mistakes.',
      mathIntuition: 'If you ask 100 people to guess the number of jellybeans in a jar, the average of their guesses is usually closer to the truth than any single person\'s guess. Random Forest applies this principle to classification.',
      realWorldExamples: [
        'Predicting whether a bank customer will default on a loan.',
        'Classifying satellite images into land-use categories (forest, urban, water).',
        'Identifying genes associated with a specific disease.'
      ]
    }
  },
  {
    id: 'svm',
    name: 'Support Vector Machine (SVM)',
    type: 'Classification',
    description: 'Finds the hyperplane that best separates different classes in the feature space.',
    whenToUse: 'When you have high-dimensional data and clear margins of separation.',
    howItWorks: [
      'Finds the "widest street" (margin) between classes.',
      'Identifies "support vectors" which are the points closest to the boundary.',
      'Can use "kernels" to handle non-linear boundaries by projecting data into higher dimensions.'
    ],
    mathIntuition: 'Maximizes the distance between the hyperplane and the nearest data points.',
    realWorldCase: 'Image recognition (e.g., identifying handwritten digits).',
    hyperparameters: [
      { id: 'C', name: 'Regularization (C)', min: 0.1, max: 10, step: 0.1, default: 1, description: 'Penalty for misclassification (lower = smoother boundary, higher = fits training data better).' }
    ],
    educationalDetails: {
      whatItDoes: 'SVM is a powerful classifier that works by finding the optimal boundary (hyperplane) between classes. It is particularly effective in high-dimensional spaces where the number of features is large.',
      howItWorks: 'It doesn\'t just look for *any* boundary; it looks for the one that has the largest "margin" (the distance to the nearest points of either class). These nearest points are called "Support Vectors" because they are the only ones that actually define the boundary.',
      mathIntuition: 'Imagine two groups of people in a field. SVM tries to drive a very wide truck between them. The wider the truck can be without hitting anyone, the more confident we are in the separation.',
      realWorldExamples: [
        'Text categorization (e.g., news vs. sports).',
        'Face detection in photos.',
        'Bioinformatics (classifying proteins or DNA sequences).'
      ]
    }
  },
  // Clustering
  {
    id: 'kmeans',
    name: 'K-Means',
    type: 'Clustering',
    description: 'Partitions n observations into k clusters in which each observation belongs to the cluster with the nearest mean.',
    whenToUse: 'When you want to group similar data points together without pre-defined labels.',
    howItWorks: [
      'Randomly initializes "K" cluster centers (centroids).',
      'Assigns each point to the nearest centroid.',
      'Recalculates centroids based on the mean of assigned points and repeats until stable.'
    ],
    mathIntuition: 'Minimizes the "Within-Cluster Sum of Squares" (WCSS).',
    realWorldCase: 'Customer segmentation for targeted marketing.',
    hyperparameters: [
      { id: 'k', name: 'K (Clusters)', min: 2, max: 10, step: 1, default: 3, description: 'The number of clusters to form.' }
    ],
    educationalDetails: {
      whatItDoes: 'K-Means is an "Unsupervised" algorithm, meaning it finds patterns in data that has no labels. It groups data points into "K" distinct clusters based on their similarity.',
      howItWorks: 'It starts by picking K random spots as "centers". It then assigns every data point to its nearest center. After that, it moves each center to the actual middle of its assigned points. It repeats this "assign and move" process until the centers stop moving.',
      mathIntuition: 'It\'s like finding the "gravity centers" of the data. The algorithm tries to make each cluster as compact as possible by minimizing the distance from points to their respective centers.',
      realWorldExamples: [
        'Grouping similar news articles together.',
        'Compressing images by reducing the number of colors used.',
        'Identifying geographical regions with similar crime patterns.'
      ]
    }
  },
  {
    id: 'hierarchical-clustering',
    name: 'Hierarchical Clustering',
    type: 'Clustering',
    description: 'Builds a hierarchy of clusters either by merging small clusters (agglomerative) or splitting large ones (divisive).',
    whenToUse: 'When you want to understand the nested structure of your data groups.',
    howItWorks: [
      'Starts with each point as its own cluster.',
      'Repeatedly merges the two closest clusters.',
      'Creates a tree-like structure called a dendrogram.'
    ],
    mathIntuition: 'Uses linkage criteria (like Ward\'s method) to determine cluster distance.',
    realWorldCase: 'Building a taxonomy of animal species based on genetic traits.',
    educationalDetails: {
      whatItDoes: 'Hierarchical Clustering creates a multi-level grouping of data. Unlike K-Means, you don\'t have to decide the number of clusters upfront. You can see how points merge together at different scales.',
      howItWorks: 'The most common version (Agglomerative) starts with every point in its own tiny cluster. It then finds the two closest clusters and merges them. This continues until everything is in one giant cluster. The history of these merges is stored in a "Dendrogram" (a tree diagram).',
      mathIntuition: 'Think of it like a family tree, but for data. You can "cut" the tree at any height to get a specific number of clusters.',
      realWorldExamples: [
        'Organizing a library of books into genres and sub-genres.',
        'Grouping cities into metropolitan areas and then into states.',
        'Analyzing social networks to find communities and sub-communities.'
      ]
    }
  },
  {
    id: 'dbscan',
    name: 'DBSCAN',
    type: 'Clustering',
    description: 'A density-based clustering non-parametric algorithm: given a set of points, it groups together points that are closely packed together.',
    whenToUse: 'When your clusters have irregular shapes or you want to identify outliers/noise.',
    howItWorks: [
      'Identifies "core points" that have many neighbors nearby.',
      'Expands clusters by connecting reachable core points.',
      'Labels points in low-density areas as noise (outliers).'
    ],
    mathIntuition: 'Groups points based on density reachability within a radius (epsilon).',
    realWorldCase: 'Identifying high-traffic zones in a city using GPS coordinates.',
    educationalDetails: {
      whatItDoes: 'DBSCAN stands for "Density-Based Spatial Clustering of Applications with Noise". It is excellent at finding clusters of any shape and is one of the few algorithms that can explicitly identify "Outliers" (noise).',
      howItWorks: 'It looks for "crowded" areas. If a point has enough neighbors within a certain distance (Epsilon), it starts a cluster. It then adds all the neighbors to that cluster, and their neighbors, and so on. Points that aren\'t part of any "crowd" are marked as noise.',
      mathIntuition: 'Imagine a party in a large park. Clusters are the groups of people standing close together talking. The people walking alone between groups are the "noise".',
      realWorldExamples: [
        'Detecting anomalies in temperature sensor data.',
        'Clustering stars and galaxies in astronomical data.',
        'Grouping GPS pings to identify popular tourist spots.'
      ]
    }
  }
];

export const DATASETS: Dataset[] = [];
