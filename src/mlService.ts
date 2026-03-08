import { Matrix } from 'ml-matrix';
import MLR from 'ml-regression-multivariate-linear';
import KNN from 'ml-knn';
import { DecisionTreeClassifier } from 'ml-cart';
import { RandomForestClassifier } from 'ml-random-forest';
import { kmeans } from 'ml-kmeans';
import { agnes } from 'ml-hclust';
import LogisticRegression from 'ml-logistic-regression';
import SVM from 'ml-svm';

const CLUSTERING_ALGORITHMS = ['kmeans', 'hierarchical-clustering', 'dbscan'];

export interface TrainingResult {
  model: any;
  type: string;
  targetMap?: Record<string, number>;
  reverseTargetMap?: Record<number, string>;
  features: string[];
  algorithmId: string;
  scalingParams: { means: number[], stds: number[] } | null;
  clusters?: number[];
  centroids?: any[];
}

export const trainModel = (
  algorithmId: string,
  data: any[],
  features: string[],
  target?: string,
  options: any = {}
): TrainingResult => {
  if (!data || data.length === 0) {
    throw new Error('Training data must not be empty');
  }
  let X = data.map(row => features.map(f => {
    const val = parseFloat(row[f]);
    return isNaN(val) ? 0 : val;
  }));
  if (X.length === 0 || X[0].length === 0) {
    throw new Error('Training features must not be empty');
  }
  
  // Feature Scaling
  let scalingParams: { means: number[], stds: number[] } | null = null;
  if (options.scaling) {
    const means = features.map((_, i) => X.reduce((sum, row) => sum + row[i], 0) / X.length);
    const stds = features.map((_, i) => {
      const variance = X.reduce((sum, row) => sum + Math.pow(row[i] - means[i], 2), 0) / X.length;
      return Math.sqrt(variance) || 1; // Avoid division by zero
    });
    X = X.map(row => row.map((val, i) => (val - means[i]) / stds[i]));
    scalingParams = { means, stds };
  }

  const isClustering = CLUSTERING_ALGORITHMS.includes(algorithmId);

  if (isClustering) {
    let result: any;
    if (algorithmId === 'kmeans') {
      const k = options.k || 3;
      if (X.length > 0 && X[0].length > 0) {
        result = kmeans(X, k, {});
      } else {
        result = { clusters: new Array(X.length).fill(0), centroids: [] };
      }
    } else if (algorithmId === 'hierarchical-clustering') {
      if (X.length > 0 && X[0].length > 0) {
        const cluster = agnes(X, {
          method: 'ward'
        });
        const k = options.k || 3;
        const root = cluster.group(k);
        const labels = new Array(X.length).fill(0);
        root.children.forEach((child, clusterIndex) => {
          child.indices().forEach(pointIndex => {
            labels[pointIndex] = clusterIndex;
          });
        });
        // Calculate centroids for prediction
        const centroids = new Array(k).fill(0).map(() => new Array(features.length).fill(0));
        const counts = new Array(k).fill(0);
        labels.forEach((clusterIdx, pointIdx) => {
          counts[clusterIdx]++;
          features.forEach((_, fIdx) => {
            centroids[clusterIdx][fIdx] += X[pointIdx][fIdx];
          });
        });
        centroids.forEach((centroid, i) => {
          if (counts[i] > 0) {
            centroid.forEach((_, j) => centroid[j] /= counts[i]);
          }
        });

        result = {
          clusters: labels,
          centroids: centroids,
          algorithmId: 'hierarchical-clustering'
        };
      } else {
        result = { clusters: new Array(X.length).fill(0), centroids: [] };
      }
    } else {
      // Placeholder for DBSCAN
      const k = options.k || 3;
      if (X.length > 0 && X[0].length > 0) {
        result = kmeans(X, k, {});
      } else {
        result = { clusters: new Array(X.length).fill(0), centroids: [] };
      }
    }

    return {
      model: result,
      type: 'clustering',
      clusters: result.clusters,
      centroids: result.centroids,
      algorithmId: algorithmId,
      features,
      scalingParams
    };
  }

  if (!target) throw new Error('Target is required for supervised learning');
  
  // Handle categorical target for classification
  const targetValues = data.map(row => row[target]);
  const uniqueTargets = Array.from(new Set(targetValues));
  const isClassification = ['logistic-regression', 'knn', 'decision-tree', 'random-forest', 'svm'].includes(algorithmId);
  
  let y: any[];
  let targetMap: Record<string, number> = {};
  let reverseTargetMap: Record<number, string> = {};

  if (isClassification) {
    y = targetValues.map(val => {
      const stringVal = val === null || val === undefined ? 'undefined' : String(val);
      if (targetMap[stringVal] === undefined) {
        const id = Object.keys(targetMap).length;
        targetMap[stringVal] = id;
        reverseTargetMap[id] = stringVal;
      }
      return targetMap[stringVal];
    });
  } else {
    y = targetValues.map(val => {
      const parsed = parseFloat(val);
      return isNaN(parsed) ? 0 : parsed;
    });
  }

  let model: any;

  switch (algorithmId) {
    case 'logistic-regression':
      model = new LogisticRegression({
        numSteps: options.numSteps || 1000,
        learningRate: options.learningRate || 0.01
      });
      if (X.length > 0 && X[0].length > 0 && y.length > 0) {
        model.train(new Matrix(X), Matrix.columnVector(y));
        // Expose weights for visualization
        if (model.classifiers && model.classifiers[0] && model.classifiers[0].weights) {
          (model as any).weights = model.classifiers[0].weights.to1DArray();
        } else {
          (model as any).weights = new Array(X[0].length).fill(0);
        }
      }
      break;
    case 'linear-regression':
    case 'multiple-linear-regression':
      if (X.length > 0 && X[0].length > 0 && y.length > 0) {
        model = new MLR(X, y.map(val => [val]));
      } else {
        model = { predict: () => [0] };
      }
      break;
    case 'polynomial-regression':
      // Simple polynomial expansion
      const degree = options.degree || 2;
      const polyX = X.map(row => {
        const newRow = [...row];
        for (let d = 2; d <= degree; d++) {
          row.forEach(val => newRow.push(Math.pow(val, d)));
        }
        return newRow;
      });
      if (polyX.length > 0 && polyX[0].length > 0 && y.length > 0) {
        model = new MLR(polyX, y.map(val => [val]));
        // Wrap model to handle prediction expansion
        const originalPredict = model.predict.bind(model);
        model.predict = (input: number[][]) => {
          const expandedInput = input.map(row => {
            const newRow = [...row];
            for (let d = 2; d <= degree; d++) {
              row.forEach(val => newRow.push(Math.pow(val, d)));
            }
            return newRow;
          });
          return originalPredict(expandedInput);
        };
      } else {
        model = { predict: () => [0] };
      }
      break;
    case 'knn':
      if (X.length > 0 && X[0].length > 0 && y.length > 0) {
        model = new KNN(X, y, { k: options.k || 3 });
      } else {
        model = { predict: () => [0] };
      }
      break;
    case 'decision-tree':
      model = new DecisionTreeClassifier({
        maxDepth: options.maxDepth || 10,
        minSamplesLeaf: options.minSamplesLeaf || 1
      });
      if (X.length > 0 && X[0].length > 0 && y.length > 0) {
        model.train(X, y);
      }
      break;
    case 'random-forest':
      try {
        const nEstimators = Math.max(1, parseInt(String(options.nEstimators)) || 20);
        const maxDepth = Math.max(1, parseInt(String(options.maxDepth)) || 10);
        model = new RandomForestClassifier({
          nEstimators,
          treeOptions: {
            maxDepth,
            minSamplesLeaf: options.minSamplesLeaf || 1
          }
        });
        if (X.length > 0 && X[0].length > 0 && y.length > 0) {
          // ml-random-forest works best with Matrix or well-formatted arrays
          model.train(new Matrix(X), y);
        } else {
          model = { predict: (input: number[][]) => input.map(() => 0), featureImportance: () => new Array(features.length).fill(0) };
        }
      } catch (e) {
        console.error('Random Forest training failed:', e);
        model = { predict: (input: number[][]) => input.map(() => 0), featureImportance: () => new Array(features.length).fill(0) };
      }
      break;
    case 'svm':
      // ml-svm is binary, so we implement One-Vs-All for multi-class support
      const svmClasses = Array.from(new Set(y));
      const classifiers = svmClasses.map(cls => {
        const svm = new SVM({
          C: options.C || 1,
          kernel: options.kernel || 'linear',
          ...options
        });
        const binaryY = y.map(val => val === cls ? 1 : -1);
        if (X.length > 0 && X[0].length > 0 && binaryY.length > 0) {
          svm.train(X, binaryY);
        }
        return { cls, svm };
      });
      
      const supportVectors: any[] = [];
      classifiers.forEach(({ svm }) => {
        const sv = (svm as any).supportVectors;
        if (Array.isArray(sv)) {
          supportVectors.push(...sv);
        }
      });

      model = {
        predict: (input: number[][]) => {
          return input.map(sample => {
            let bestClass = svmClasses[0];
            let maxMargin = -Infinity;
            classifiers.forEach(({ cls, svm }) => {
              const margin = svm.margin(sample);
              if (margin > maxMargin) {
                maxMargin = margin;
                bestClass = cls;
              }
            });
            return bestClass;
          });
        },
        classifiers,
        supportVectors
      };
      break;
    default:
      // Fallback or placeholder for unimplemented ones
      if (X.length > 0 && X[0].length > 0 && y.length > 0) {
        model = new MLR(X, y.map(val => [val]));
      } else {
        model = { predict: () => [0] };
      }
  }

  return {
    model,
    type: isClassification ? 'classification' : 'regression',
    targetMap,
    reverseTargetMap,
    features,
    algorithmId,
    scalingParams
  };
};

export const crossValidate = (
  algorithmId: string,
  data: any[],
  features: string[],
  target: string,
  options: any = {},
  kFolds: number = 5
) => {
  if (!Array.isArray(data)) return { avgMetrics: {}, folds: [] };
  const shuffled = [...data].sort(() => Math.random() - 0.5);
  const foldSize = Math.floor(shuffled.length / kFolds);
  const results = [];

  for (let i = 0; i < kFolds; i++) {
    const testStart = i * foldSize;
    const testEnd = testStart + foldSize;
    const testData = shuffled.slice(testStart, testEnd);
    const trainData = [...shuffled.slice(0, testStart), ...shuffled.slice(testEnd)];

    const model = trainModel(algorithmId, trainData, features, target, options);
    const metrics = evaluate(model, testData, features, target);
    results.push(metrics);
  }

  // Average metrics
  const avgMetrics: any = {};
  if (results.length === 0) return { avgMetrics, folds: [] };
  const keys = Object.keys(results[0]).filter(k => typeof results[0][k] === 'number');
  keys.forEach(key => {
    avgMetrics[key] = results.reduce((sum, r) => sum + r[key], 0) / kFolds;
  });

  return { avgMetrics, folds: results };
};

export const generateROCData = (trainedModel: any, testData: any[], features: string[], target: string) => {
  if (trainedModel.type !== 'classification') return null;
  if (!Array.isArray(testData)) return null;
  
  const classes = Object.keys(trainedModel.targetMap);
  if (classes.length !== 2) return null; // ROC is typically for binary

  const X_test = testData.map(row => features.map(f => parseFloat(row[f])));
  const y_test = testData.map(row => trainedModel.targetMap[row[target]]);

  // Get probabilities if supported
  const probs = X_test.map(x => {
    const { model, algorithmId, scalingParams } = trainedModel;
    let processedX = [...x];
    if (scalingParams) {
      processedX = processedX.map((val, i) => (val - scalingParams.means[i]) / scalingParams.stds[i]);
    }

    if (algorithmId === 'logistic-regression') {
      const logit = processedX.reduce((sum, val, i) => sum + val * model.weights[i], 0);
      return 1 / (1 + Math.exp(-logit));
    }
    // Fallback for others: use predict as 0 or 1
    return predict(trainedModel, x) === trainedModel.reverseTargetMap[1] ? 1 : 0;
  });

  const thresholds = Array.from({ length: 21 }, (_, i) => i / 20);
  return thresholds.map(t => {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    probs.forEach((p, i) => {
      const pred = p >= t ? 1 : 0;
      const actual = y_test[i];
      if (pred === 1 && actual === 1) tp++;
      else if (pred === 1 && actual === 0) fp++;
      else if (pred === 0 && actual === 0) tn++;
      else if (pred === 0 && actual === 1) fn++;
    });
    const tpr = tp / (tp + fn) || 0;
    const fpr = fp / (fp + tn) || 0;
    return { fpr, tpr, threshold: t };
  }).sort((a, b) => a.fpr - b.fpr);
};

export const generateLearningCurveData = (
  algorithmId: string,
  data: any[],
  features: string[],
  target: string,
  options: any = {}
) => {
  if (!Array.isArray(data)) return [];
  const shuffled = [...data].sort(() => Math.random() - 0.5);
  const testSize = Math.floor(shuffled.length * 0.2);
  const testData = shuffled.slice(0, testSize);
  const remainingData = shuffled.slice(testSize);

  const sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0].map(p => Math.max(5, Math.floor(remainingData.length * p)));
  
  return sizes.map(size => {
    const trainSubset = remainingData.slice(0, size);
    
    // Use 3-fold CV for each size to get more stable error estimates
    const kFolds = 3;
    const foldSize = Math.floor(trainSubset.length / kFolds);
    let totalTrainError = 0;
    let totalValError = 0;

    for (let i = 0; i < kFolds; i++) {
      const valStart = i * foldSize;
      const valEnd = valStart + foldSize;
      const valData = trainSubset.slice(valStart, valEnd);
      const subTrainData = [...trainSubset.slice(0, valStart), ...trainSubset.slice(valEnd)];

      if (subTrainData.length < 2) continue;

      const model = trainModel(algorithmId, subTrainData, features, target, options);
      const trainMetrics = evaluate(model, subTrainData, features, target);
      const valMetrics = evaluate(model, valData.length > 0 ? valData : testData, features, target);
      
      if (model.type === 'classification') {
        totalTrainError += (1 - (trainMetrics.accuracy || 0));
        totalValError += (1 - (valMetrics.accuracy || 0));
      } else {
        totalTrainError += (trainMetrics.mse || 0);
        totalValError += (valMetrics.mse || 0);
      }
    }
    
    return { 
      size, 
      trainError: totalTrainError / kFolds, 
      testError: totalValError / kFolds 
    };
  });
};

export const generateBiasVarianceData = (
  algorithmId: string,
  data: any[],
  features: string[],
  target: string,
  paramName: string,
  paramValues: number[]
) => {
  if (!Array.isArray(data)) return [];
  const shuffled = [...data].sort(() => Math.random() - 0.5);
  const splitIdx = Math.floor(shuffled.length * 0.8);
  const trainData = shuffled.slice(0, splitIdx);
  const testData = shuffled.slice(splitIdx);

  return paramValues.map(val => {
    const options = { [paramName]: val };
    const model = trainModel(algorithmId, trainData, features, target, options);
    const trainMetrics = evaluate(model, trainData, features, target);
    const testMetrics = evaluate(model, testData, features, target);
    
    const isClassification = model.type === 'classification';
    
    return { 
      complexity: val, 
      trainError: isClassification ? 1 - (trainMetrics.accuracy || 0) : (trainMetrics.mse || 0), 
      testError: isClassification ? 1 - (testMetrics.accuracy || 0) : (testMetrics.mse || 0) 
    };
  });
};

export const getTreeStructure = (model: any) => {
  if (!model || !model.root) return null;
  
  const formatNode = (node: any): any => {
    if (!node) return null;
    if (node.splitColumn === undefined) {
      return { name: `Class: ${node.distribution ? Object.keys(node.distribution)[0] : 'Leaf'}` };
    }
    return {
      name: `Feature ${node.splitColumn} <= ${node.splitValue.toFixed(2)}`,
      children: [
        formatNode(node.left),
        formatNode(node.right)
      ].filter(Boolean)
    };
  };
  
  return formatNode(model.root);
};

export const trainKMeansIterative = (
  data: number[][],
  k: number,
  maxIterations: number = 10
) => {
  let centroids = data.slice(0, k).map(p => [...p]);
  const history = [{ centroids: centroids.map(c => [...c]), clusters: [] as number[] }];

  for (let iter = 0; iter < maxIterations; iter++) {
    const clusters = data.map(point => {
      let minDist = Infinity;
      let clusterIdx = 0;
      centroids.forEach((c, i) => {
        const dist = Math.sqrt(point.reduce((sum, val, j) => sum + Math.pow(val - c[j], 2), 0));
        if (dist < minDist) {
          minDist = dist;
          clusterIdx = i;
        }
      });
      return clusterIdx;
    });

    const newCentroids = Array.from({ length: k }, () => new Array(data[0].length).fill(0));
    const counts = new Array(k).fill(0);

    clusters.forEach((c, i) => {
      counts[c]++;
      data[i].forEach((val, j) => newCentroids[c][j] += val);
    });

    newCentroids.forEach((c, i) => {
      if (counts[i] > 0) c.forEach((_, j) => c[j] /= counts[i]);
      else newCentroids[i] = centroids[i]; // Keep old if empty
    });

    centroids = newCentroids;
    history.push({ centroids: centroids.map(c => [...c]), clusters });
    
    // Check convergence
    const diff = history[history.length - 2].centroids.reduce((sum, c, i) => 
      sum + Math.sqrt(c.reduce((s, v, j) => s + Math.pow(v - centroids[i][j], 2), 0)), 0);
    if (diff < 0.001) break;
  }

  return history;
};

export const predict = (trainedModel: any, input: number[]) => {
  const { model, type, reverseTargetMap, algorithmId, centroids, scalingParams } = trainedModel;
  
  // Apply scaling to input if it was used during training
  let processedInput = [...input];
  if (scalingParams) {
    const { means, stds } = scalingParams;
    processedInput = processedInput.map((val, i) => (val - means[i]) / stds[i]);
  }

  if (CLUSTERING_ALGORITHMS.includes(algorithmId)) {
    // Robust centroid-based prediction for all clustering algorithms
    if (centroids && centroids.length > 0) {
      let minDistance = Infinity;
      let nearestCluster = 0;
      
      centroids.forEach((centroid: any, index: number) => {
        // Handle both KMeansResult centroids and custom centroids
        const c = Array.isArray(centroid) ? centroid : (centroid.centroid || []);
        if (!c || c.length === 0) return;
        
        let distance = 0;
        for (let i = 0; i < processedInput.length; i++) {
          distance += Math.pow(processedInput[i] - c[i], 2);
        }
        if (distance < minDistance) {
          minDistance = distance;
          nearestCluster = index;
        }
      });
      return nearestCluster;
    }
    
    return 0;
  }

  if (type === 'classification') {
    let prediction;
    if (algorithmId === 'logistic-regression') {
      if (processedInput.length === 0) return 0;
      prediction = model.predict(new Matrix([processedInput]))[0];
    } else if (algorithmId === 'svm') {
      // Our custom SVM wrapper returns an array for predict
      prediction = model.predict([processedInput])[0];
    } else {
      prediction = model.predict([processedInput])[0];
    }
    return reverseTargetMap ? reverseTargetMap[prediction] : prediction;
  } else {
    const prediction = model.predict([processedInput])[0];
    return Array.isArray(prediction) ? prediction[0] : prediction;
  }
};

export const evaluate = (trainedModel: any, testData: any[], features: string[], target: string) => {
  if (!testData || testData.length === 0) {
    return trainedModel.type === 'classification' 
      ? { accuracy: 0, precision: 0, recall: 0, f1: 0, confusionMatrix: [], classes: [] }
      : { mse: 0, mae: 0, r2: 0 };
  }
  const X_test = testData.map(row => features.map(f => parseFloat(row[f])));
  
  if (trainedModel.type === 'clustering') {
    // For clustering, we can calculate inertia (Within-Cluster Sum of Squares)
    const centroids = trainedModel.centroids;
    let inertia = 0;
    
    X_test.forEach(x => {
      const clusterIdx = predict(trainedModel, x);
      const centroid = centroids[clusterIdx];
      if (centroid) {
        let distSq = 0;
        for (let i = 0; i < x.length; i++) {
          distSq += Math.pow(x[i] - (Array.isArray(centroid) ? centroid[i] : centroid.centroid[i]), 2);
        }
        inertia += distSq;
      }
    });
    
    return { inertia };
  }

  const y_test = testData.map(row => row[target]);
  if (y_test.length === 0) {
    return trainedModel.type === 'classification' 
      ? { accuracy: 0, precision: 0, recall: 0, f1: 0, confusionMatrix: [], classes: [] }
      : { mse: 0, mae: 0, r2: 0 };
  }
  
  const predictions = X_test.map(x => predict(trainedModel, x));
  
  if (trainedModel.type === 'classification') {
    let correct = 0;
    predictions.forEach((p, i) => {
      if (p === y_test[i]) correct++;
    });
    const accuracy = correct / y_test.length;
    
    // Calculate Precision, Recall, F1
    const classes = Object.keys(trainedModel.targetMap);
    const metricsByClass = classes.map(cls => {
      const actualPositives = y_test.filter(y => y === cls).length;
      const predictedPositives = predictions.filter(p => p === cls).length;
      const truePositives = predictions.filter((p, i) => p === cls && y_test[i] === cls).length;
      
      const precision = predictedPositives === 0 ? 0 : truePositives / predictedPositives;
      const recall = actualPositives === 0 ? 0 : truePositives / actualPositives;
      const f1 = (precision + recall) === 0 ? 0 : 2 * (precision * recall) / (precision + recall);
      
      return { cls, precision, recall, f1 };
    });

    const avgPrecision = metricsByClass.reduce((a, b) => a + b.precision, 0) / classes.length;
    const avgRecall = metricsByClass.reduce((a, b) => a + b.recall, 0) / classes.length;
    const avgF1 = metricsByClass.reduce((a, b) => a + b.f1, 0) / classes.length;

    // Simple confusion matrix
    const matrix = classes.map(actual => {
      return classes.map(pred => {
        return predictions.filter((p, i) => p === pred && y_test[i] === actual).length;
      });
    });

    return { 
      accuracy, 
      precision: avgPrecision, 
      recall: avgRecall, 
      f1: avgF1, 
      confusionMatrix: matrix, 
      classes 
    };
  } else {
    // Regression metrics
    let sumSquaredError = 0;
    let sumAbsoluteError = 0;
    const y_mean = y_test.reduce((a, b) => a + b, 0) / y_test.length;
    let totalSumSquares = 0;

    y_test.forEach((actual, i) => {
      const pred = predictions[i];
      const error = actual - pred;
      sumSquaredError += error * error;
      sumAbsoluteError += Math.abs(error);
      totalSumSquares += (actual - y_mean) * (actual - y_mean);
    });

    const mse = sumSquaredError / y_test.length;
    const mae = sumAbsoluteError / y_test.length;
    const r2 = 1 - (sumSquaredError / totalSumSquares);

    return { mse, mae, r2 };
  }
};

export const getFeatureImportance = (trainedModel: any) => {
  const { model, algorithmId, features } = trainedModel;
  if (!model) return null;

  let importances: number[] = [];

  if (algorithmId === 'random-forest') {
    try {
      let rawImportances;
      if (typeof model.featureImportance === 'function') {
        rawImportances = model.featureImportance();
      } else if (typeof model.getFeatureImportance === 'function') {
        rawImportances = model.getFeatureImportance();
      }

      // If built-in returns zeros or is not an array, try manual calculation from individual trees
      if (Array.isArray(rawImportances) && rawImportances.some(v => v > 0)) {
        importances = rawImportances;
      } else {
        importances = new Array(features.length).fill(0);
        const trees = model.models || model.estimators || [];
        if (Array.isArray(trees) && trees.length > 0) {
          trees.forEach((tree: any) => {
            const traverse = (node: any) => {
              if (!node || node.splitColumn === undefined) return;
              // Use gain if available, otherwise use 1 as a proxy for split count
              const weight = (typeof node.gain === 'number' && node.gain > 0) ? node.gain : 1;
              importances[node.splitColumn] += weight;
              traverse(node.left);
              traverse(node.right);
            };
            traverse(tree.root);
          });
          
          const sum = importances.reduce((a, b) => a + b, 0);
          if (sum > 0) {
            importances = importances.map(v => v / sum);
          }
        }
      }
    } catch (e) {
      console.error('Failed to get Random Forest importance:', e);
      importances = new Array(features.length).fill(0);
    }
  } else if (algorithmId === 'decision-tree') {
    // Decision tree importance is simpler, but we can extract it
    importances = new Array(features.length).fill(0);
    const traverse = (node: any) => {
      if (!node || node.splitColumn === undefined) return;
      importances[node.splitColumn] += (node.gain || 0);
      traverse(node.left);
      traverse(node.right);
    };
    traverse(model.root);
    const sum = importances.reduce((a, b) => a + b, 0);
    if (sum > 0) importances = importances.map(v => v / sum);
  } else {
    return null;
  }

  return features.map((name, i) => ({ name, value: importances[i] || 0 }))
    .sort((a, b) => b.value - a.value);
};

export const getDecisionBoundary = (trainedModel: any, data: any[], features: string[]) => {
  if (features.length !== 2) return null;

  const xMin = Math.min(...data.map(d => parseFloat(d[features[0]])));
  const xMax = Math.max(...data.map(d => parseFloat(d[features[0]])));
  const yMin = Math.min(...data.map(d => parseFloat(d[features[1]])));
  const yMax = Math.max(...data.map(d => parseFloat(d[features[1]])));

  const xRange = xMax - xMin;
  const yRange = yMax - yMin;
  
  const padding = 0.1;
  const xStart = xMin - xRange * padding;
  const xEnd = xMax + xRange * padding;
  const yStart = yMin - yRange * padding;
  const yEnd = yMax + yRange * padding;

  const resolution = 30;
  const grid = [];

  for (let i = 0; i <= resolution; i++) {
    for (let j = 0; j <= resolution; j++) {
      const x = xStart + (xEnd - xStart) * (i / resolution);
      const y = yStart + (yEnd - yStart) * (j / resolution);
      const pred = predict(trainedModel, [x, y]);
      
      let margin = 0;
      if (trainedModel.algorithmId === 'svm' && trainedModel.model.classifiers) {
        // For SVM, we can show the margin of the first classifier (binary case)
        margin = trainedModel.model.classifiers[0].svm.margin([x, y]);
      }
      
      grid.push({ x, y, pred, margin });
    }
  }

  return { grid, xRange: [xStart, xEnd], yRange: [yStart, yEnd] };
};
