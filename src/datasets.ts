import { Dataset } from './types';

// Helper to generate sample data for classification (Iris-like)
const generateIrisData = () => {
  const species = ['setosa', 'versicolor', 'virginica'];
  const data = [];
  for (let i = 0; i < 150; i++) {
    const s = species[Math.floor(i / 50)];
    const base = i / 50;
    data.push({
      sepal_length: (5.0 + base + Math.random() * 0.5).toFixed(1),
      sepal_width: (3.0 - base * 0.2 + Math.random() * 0.5).toFixed(1),
      petal_length: (1.5 + base * 2 + Math.random() * 0.5).toFixed(1),
      petal_width: (0.2 + base * 0.8 + Math.random() * 0.3).toFixed(1),
      species: s
    });
  }
  return data;
};

// Helper to generate sample data for regression (Diabetes-like)
const generateDiabetesData = () => {
  const data = [];
  for (let i = 0; i < 100; i++) {
    const bmi = 20 + Math.random() * 15;
    const bp = 80 + Math.random() * 40;
    const age = 20 + Math.random() * 60;
    data.push({
      age: parseFloat(age.toFixed(1)),
      bmi: parseFloat(bmi.toFixed(1)),
      bp: parseFloat(bp.toFixed(1)),
      progression: Math.floor(bmi * 5 + bp * 2 + age * 0.5 + Math.random() * 20)
    });
  }
  return data;
};

// Helper to generate blobs for clustering
const generateBlobData = () => {
  const data = [];
  const centers = [[2, 2], [8, 8], [2, 8]];
  for (let i = 0; i < 90; i++) {
    const center = centers[i % 3];
    data.push({
      x: parseFloat((center[0] + (Math.random() - 0.5) * 3).toFixed(2)),
      y: parseFloat((center[1] + (Math.random() - 0.5) * 3).toFixed(2))
    });
  }
  return data;
};

export const FULL_DATASETS: Dataset[] = [
  {
    id: 'iris',
    name: 'Iris Flower',
    description: 'Classify iris flowers into 3 species based on petal and sepal dimensions.',
    type: 'classification',
    features: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    target: 'species',
    columnDescriptions: {
      sepal_length: 'Length of the sepal (outer part of the flower) in cm.',
      sepal_width: 'Width of the sepal in cm.',
      petal_length: 'Length of the petal (inner part of the flower) in cm.',
      petal_width: 'Width of the petal in cm.',
      species: 'The specific species of Iris flower (Setosa, Versicolor, or Virginica).'
    },
    data: generateIrisData()
  },
  {
    id: 'breast_cancer',
    name: 'Breast Cancer',
    description: 'Predict whether a tumor is malignant or benign based on cell characteristics.',
    type: 'classification',
    features: ['radius', 'texture', 'perimeter', 'area', 'smoothness'],
    target: 'diagnosis',
    columnDescriptions: {
      radius: 'Mean of distances from center to points on the perimeter.',
      texture: 'Standard deviation of gray-scale values.',
      perimeter: 'The distance around the outside of the cell nucleus.',
      area: 'The total surface area of the cell nucleus.',
      smoothness: 'Local variation in radius lengths.',
      diagnosis: 'The outcome: Malignant (cancerous) or Benign (non-cancerous).'
    },
    data: Array.from({ length: 100 }, () => ({
      radius: parseFloat((10 + Math.random() * 10).toFixed(2)),
      texture: parseFloat((15 + Math.random() * 10).toFixed(2)),
      perimeter: parseFloat((60 + Math.random() * 60).toFixed(2)),
      area: parseFloat((300 + Math.random() * 1000).toFixed(2)),
      smoothness: parseFloat((0.05 + Math.random() * 0.1).toFixed(3)),
      diagnosis: Math.random() > 0.5 ? 'malignant' : 'benign'
    }))
  },
  {
    id: 'diabetes',
    name: 'Diabetes Progression',
    description: 'Predict disease progression one year after baseline.',
    type: 'regression',
    features: ['age', 'bmi', 'bp'],
    target: 'progression',
    columnDescriptions: {
      age: 'Age of the patient in years.',
      bmi: 'Body Mass Index, a measure of body fat based on height and weight.',
      bp: 'Average blood pressure of the patient.',
      progression: 'A quantitative measure of disease progression one year after baseline.'
    },
    data: generateDiabetesData()
  },
  {
    id: 'housing',
    name: 'California Housing',
    description: 'Predict median house value based on location and demographics.',
    type: 'regression',
    features: ['income', 'house_age', 'rooms', 'bedrooms', 'population'],
    target: 'price',
    columnDescriptions: {
      income: 'Median income for households within a block (in tens of thousands of US Dollars).',
      house_age: 'Median age of a house within a block.',
      rooms: 'Total number of rooms within a block.',
      bedrooms: 'Total number of bedrooms within a block.',
      population: 'Total number of people residing within a block.',
      price: 'Median house value for households within a block (in US Dollars).'
    },
    data: Array.from({ length: 100 }, () => ({
      income: parseFloat((2 + Math.random() * 10).toFixed(2)),
      house_age: Math.floor(5 + Math.random() * 45),
      rooms: parseFloat((3 + Math.random() * 5).toFixed(1)),
      bedrooms: parseFloat((1 + Math.random() * 2).toFixed(1)),
      population: Math.floor(100 + Math.random() * 5000),
      price: Math.floor(100000 + Math.random() * 400000)
    }))
  },
  {
    id: 'blobs',
    name: 'Synthetic Blobs',
    description: 'Group synthetic data points into clusters based on spatial proximity.',
    type: 'clustering',
    features: ['x', 'y'],
    columnDescriptions: {
      x: 'Horizontal coordinate in a 2D space.',
      y: 'Vertical coordinate in a 2D space.'
    },
    data: generateBlobData()
  }
];
