import { motion, AnimatePresence } from 'motion/react';
import React, { useState, useEffect } from 'react';
import { 
  ChevronRight, Database, Settings, Play, BarChart2, RefreshCw, HelpCircle, 
  Info, AlertTriangle, CheckCircle2, ArrowRight, Layers, Cpu, Zap, 
  TrendingUp, Target, Activity, Microscope, BookOpen, Github, RotateCcw
} from 'lucide-react';
import { ALGORITHMS, AlgorithmInfo, Dataset, Hyperparameter, HistoryItem } from './types';
import { FULL_DATASETS } from './datasets';
import { DataExploration } from './components/DataExploration';
import { ExperimentHistory } from './components/ExperimentHistory';
import { 
  trainModel, evaluate, predict, getFeatureImportance, getDecisionBoundary,
  crossValidate, generateROCData, generateLearningCurveData, generateBiasVarianceData,
  getTreeStructure, trainKMeansIterative
} from './mlService';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis, Cell, BarChart, Bar, AreaChart, Area
} from 'recharts';

export default function App() {
  const [step, setStep] = useState(0);
  const [selectedAlgo, setSelectedAlgo] = useState<AlgorithmInfo | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [features, setFeatures] = useState<string[]>([]);
  const [target, setTarget] = useState<string>('');
  const [splitRatio, setSplitRatio] = useState(0.8);
  const [scaling, setScaling] = useState(true);
  const [isCrossValidation, setIsCrossValidation] = useState(false);
  const [kFolds, setKFolds] = useState(5);
  const [hyperparams, setHyperparams] = useState<Record<string, number>>({});
  
  const [trainedModel, setTrainedModel] = useState<any>(null);
  const [evaluation, setEvaluation] = useState<any>(null);
  const [trainEvaluation, setTrainEvaluation] = useState<any>(null);
  const [cvResults, setCVResults] = useState<any>(null);
  const [featureImportance, setFeatureImportance] = useState<any>(null);
  const [decisionBoundary, setDecisionBoundary] = useState<any>(null);
  const [rocData, setROCData] = useState<any>(null);
  const [learningCurveData, setLearningCurveData] = useState<any>(null);
  const [biasVarianceData, setBiasVarianceData] = useState<any>(null);
  const [treeData, setTreeData] = useState<any>(null);
  const [kmeansHistory, setKMeansHistory] = useState<any[]>([]);
  const [currentKMeansStep, setCurrentKMeansStep] = useState(0);
  const [selectedPoint, setSelectedPoint] = useState<any>(null);
  const [knnNeighbors, setKNNNeighbors] = useState<any[]>([]);
  
  const [testInput, setTestInput] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<any>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  const [showDocumentation, setShowDocumentation] = useState(false);

  // Comparison Mode State
  const [comparisonModels, setComparisonModels] = useState<any[]>([]);
  const [isComparisonMode, setIsComparisonMode] = useState(false);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [step, showHistory, showAbout, showDocumentation]);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/history');
      const data = await res.json();
      setHistory(data);
    } catch (error) {
      console.error('Failed to fetch history:', error);
    }
  };

  const saveExperiment = async (metrics: any) => {
    if (!selectedAlgo || !selectedDataset) return;
    try {
      await fetch('/api/history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          algo_id: selectedAlgo.id,
          algo_name: selectedAlgo.name,
          dataset_id: selectedDataset.id,
          dataset_name: selectedDataset.name,
          features,
          target,
          hyperparams,
          metrics
        })
      });
      fetchHistory();
    } catch (error) {
      console.error('Failed to save experiment:', error);
    }
  };

  const deleteExperiment = async (id: number) => {
    try {
      await fetch(`/api/history/${id}`, { method: 'DELETE' });
      fetchHistory();
    } catch (error) {
      console.error('Failed to delete experiment:', error);
    }
  };

  const reRunExperiment = (item: HistoryItem) => {
    const algo = ALGORITHMS.find(a => a.id === item.algo_id);
    const dataset = FULL_DATASETS.find(d => d.id === item.dataset_id);
    if (algo && dataset) {
      setSelectedAlgo(algo);
      setSelectedDataset(dataset);
      setFeatures(item.features);
      setTarget(item.target);
      setHyperparams(item.hyperparams);
      setStep(4); // Go to evaluation
      // handleTrain will be triggered by useEffect on hyperparams if we are at step 4
    }
  };

  useEffect(() => {
    if (step === 4 && selectedAlgo && selectedDataset) {
      handleTrain();
    }
  }, [hyperparams, step]);

  const reset = () => {
    setStep(0);
    setSelectedAlgo(null);
    setSelectedDataset(null);
    setFeatures([]);
    setTarget('');
    setTrainedModel(null);
    setEvaluation(null);
    setTrainEvaluation(null);
    setPrediction(null);
    setHyperparams({});
    setScaling(true);
    setSplitRatio(0.8);
    setKFolds(5);
    setIsCrossValidation(false);
    setFeatureImportance(null);
    setDecisionBoundary(null);
    setComparisonModels([]);
    setIsComparisonMode(false);
  };

  const handleAlgoSelect = (algo: AlgorithmInfo) => {
    setSelectedAlgo(algo);
    // Set default hyperparams
    const defaults: Record<string, any> = {};
    algo.hyperparameters?.forEach(hp => {
      defaults[hp.id] = hp.default;
    });
    setHyperparams(defaults);
    setStep(1);
  };

  const handleDatasetSelect = (ds: Dataset) => {
    setSelectedDataset(ds);
    setFeatures(ds.features);
    setTarget(ds.target || '');
    setStep(2);
  };

  const handleTrain = async () => {
    if (!selectedDataset || !selectedAlgo) return;
    setIsTraining(true);
    
    try {
      // Reset previous results
      setTrainedModel(null);
      setEvaluation(null);
      setTrainEvaluation(null);
      setCVResults(null);
      setROCData(null);
      setLearningCurveData(null);
      setBiasVarianceData(null);
      setTreeData(null);
      setKMeansHistory([]);
      setCurrentKMeansStep(0);

      const data = selectedDataset.data;
      if (!Array.isArray(data)) {
        setIsTraining(false);
        return;
      }

      if (features.length === 0) {
        setIsTraining(false);
        alert('Please select at least one feature.');
        return;
      }
      const shuffled = [...data].sort(() => Math.random() - 0.5);
      const splitIndex = Math.floor(shuffled.length * splitRatio);
      const trainData = shuffled.slice(0, splitIndex);
      const testData = shuffled.slice(splitIndex);

      if (trainData.length === 0) {
        setIsTraining(false);
        alert('Training set is empty. Please check your data split ratio.');
        return;
      }

      // K-Means Animation Data
      if (selectedAlgo.id === 'kmeans') {
        const numericData = data.map(row => features.map(f => parseFloat(row[f])));
        const history = trainKMeansIterative(numericData, hyperparams.k || 3);
        setKMeansHistory(history);
        // We will animate this in Step 3
      }

      if (isCrossValidation && selectedAlgo.type !== 'Clustering') {
        const cv = crossValidate(selectedAlgo.id, data, features, target, { ...hyperparams, scaling }, kFolds);
        setCVResults(cv);
      }

      const startTime = performance.now();
      const result = trainModel(selectedAlgo.id, trainData, features, target, { ...hyperparams, scaling });
      const trainingTime = performance.now() - startTime;
      setTrainedModel(result);

      const evalResult = { ...evaluate(result, testData, features, target), trainingTime };
      const trainEvalResult = evaluate(result, trainData, features, target);
      setEvaluation(evalResult);
      setTrainEvaluation(trainEvalResult);
      
      // Save to history
      saveExperiment(evalResult);

      if (selectedAlgo.type !== 'Clustering') {
        const importance = getFeatureImportance(result);
        setFeatureImportance(importance);

        if (features.length === 2) {
          const boundary = getDecisionBoundary(result, data, features);
          setDecisionBoundary(boundary);
        }

        // Advanced Visualizations Data
        const roc = generateROCData(result, testData, features, target);
        setROCData(roc);

        const learningCurve = generateLearningCurveData(selectedAlgo.id, data, features, target, { ...hyperparams, scaling });
        setLearningCurveData(learningCurve);

        if (selectedAlgo.complexityParameter) {
          const param = selectedAlgo.complexityParameter;
          const h = selectedAlgo.hyperparameters?.find(x => x.id === param);
          if (h) {
            const values = Array.from({ length: 5 }, (_, i) => h.min + (i * (h.max - h.min) / 4));
            const bv = generateBiasVarianceData(selectedAlgo.id, data, features, target, param, values);
            setBiasVarianceData(bv);
          }
        }

        if (selectedAlgo.id === 'decision-tree') {
          const tree = getTreeStructure(result.model);
          setTreeData(tree);
        }

        // Add to comparison models
        setComparisonModels(prev => [
          ...prev,
          {
            id: Date.now(),
            algoName: selectedAlgo.name,
            datasetId: selectedDataset.id,
            datasetName: selectedDataset.name,
            metrics: evalResult,
            hyperparams: { ...hyperparams }
          }
        ]);
      }
      
      setStep(4);
      // Force scroll to top after UI update
      setTimeout(() => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }, 100);
    } catch (error) {
      console.error('Training failed:', error);
      alert('An error occurred during training. Please check your parameters and data.');
    } finally {
      setIsTraining(false);
    }
  };

  const handlePredict = () => {
    if (!trainedModel) return;
    const input = features.map(f => parseFloat(testInput[f] || '0'));
    const res = predict(trainedModel, input);
    setPrediction(res);
  };

  const getInsights = () => {
    if (!evaluation) return null;
    if (selectedAlgo?.type === 'Clustering') {
      return "Clustering complete. Review the inertia and cluster distribution to assess how well the data points are grouped.";
    }
    
    const acc = evaluation.accuracy || evaluation.r2;
    if (acc > 0.9) {
      return "Excellent performance! The model has captured the underlying patterns very well. It generalizes effectively to unseen data.";
    } else if (acc > 0.7) {
      return "Good performance. The model is reliable, but there might be room for improvement by tuning hyperparameters or adding more features.";
    } else {
      return "Performance is low. This could be due to underfitting (model too simple) or overfitting (model too complex). Try adjusting hyperparameters, scaling features, or selecting different inputs.";
    }
  };

  const getWarnings = () => {
    const warnings = [];
    if (['knn', 'svm'].includes(selectedAlgo?.id || '') && !scaling) {
      warnings.push({ id: 'scaling', text: 'Distance-based models (KNN, SVM) perform much better with feature scaling.', type: 'warning' });
    }
    if (splitRatio > 0.85) {
      warnings.push({ id: 'split', text: 'Training set is very large. This might lead to overfitting or unreliable evaluation.', type: 'warning' });
    }
    if (features.length === 1) {
      warnings.push({ id: 'features', text: 'Using only one feature may limit the model\'s ability to learn complex patterns.', type: 'info' });
    }
    return warnings;
  };

  return (
    <div className="min-h-screen bg-[#F5F5F0] text-[#1A1A1A] font-sans selection:bg-emerald-200">
      {/* Header */}
      <header className="border-b border-black/5 bg-white/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer" onClick={reset}>
            <div className="w-8 h-8 bg-emerald-600 rounded-lg flex items-center justify-center text-white font-bold">ML</div>
            <h1 className="text-xl font-semibold tracking-tight">Interactive Lab</h1>
          </div>
          <nav className="flex gap-8 items-center">
            {[
              { label: 'Explore', targetStep: 0, enabled: true },
              { label: 'Datasets', targetStep: 1, enabled: !!selectedAlgo },
              { label: 'Analysis', targetStep: 2, enabled: !!selectedDataset },
              { label: 'Pipeline', targetStep: 3, enabled: !!selectedDataset },
              { label: 'Results', targetStep: 4, enabled: !!trainedModel }
            ].map((item, i) => (
              <button 
                key={item.label} 
                disabled={!item.enabled}
                onClick={() => setStep(item.targetStep)}
                className={`text-sm font-medium transition-colors cursor-pointer disabled:cursor-not-allowed ${step === item.targetStep ? 'text-emerald-600' : item.enabled ? 'text-black/80 hover:text-emerald-500' : 'text-black/20'}`}
              >
                {item.label}
              </button>
            ))}
            <div className="w-px h-4 bg-black/10 mx-2" />
            <button 
              onClick={() => {
                setShowHistory(!showHistory);
                setShowAbout(false);
                setShowDocumentation(false);
              }}
              className={`text-sm font-bold uppercase tracking-widest transition-colors ${showHistory ? 'text-emerald-600' : 'text-black/40 hover:text-emerald-500'}`}
            >
              Notebook
            </button>
          </nav>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-12">
        {showAbout ? (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-4xl mx-auto">
            <div className="flex items-center gap-2 text-emerald-600 mb-12 cursor-pointer hover:underline" onClick={() => setShowAbout(false)}>
              <ChevronRight className="rotate-180 w-4 h-4" />
              <span className="text-sm font-medium">Back to Lab</span>
            </div>
            
            <div className="space-y-16">
              <section>
                <h2 className="text-5xl font-bold tracking-tight mb-8">About ML Interactive Lab</h2>
                <div className="prose prose-lg max-w-none text-black/70 leading-relaxed space-y-6">
                  <p className="text-xl text-black/90 font-medium">
                    ML Interactive Lab was created with a simple goal: <span className="text-emerald-600 font-bold">make machine learning easier to understand by letting students experiment with it directly</span>.
                  </p>
                  <p>
                    Many beginners struggle with machine learning because most tutorials focus heavily on theory or large blocks of code. While these resources are valuable, they often make it difficult for students to see how different parts of the machine learning pipeline actually work together.
                  </p>
                  <p>
                    ML Interactive Lab solves this problem by turning machine learning into an <span className="font-bold text-black">interactive learning experience</span>.
                  </p>
                </div>
              </section>

              <section className="bg-emerald-50 p-12 rounded-[3rem] border border-emerald-100">
                <h3 className="text-2xl font-bold mb-6 text-emerald-900">Interactive Learning Experience</h3>
                <p className="mb-8 text-emerald-900/70">Instead of only reading about algorithms, students can:</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {[
                    { icon: <Microscope className="w-5 h-5" />, text: "Select machine learning algorithms" },
                    { icon: <Database className="w-5 h-5" />, text: "Choose real datasets" },
                    { icon: <Settings className="w-5 h-5" />, text: "Select features and preprocessing techniques" },
                    { icon: <Play className="w-5 h-5" />, text: "Train models instantly" },
                    { icon: <BarChart2 className="w-5 h-5" />, text: "Evaluate performance metrics" },
                    { icon: <Target className="w-5 h-5" />, text: "Experiment with predictions" }
                  ].map((item, i) => (
                    <div key={i} className="flex items-center gap-4 bg-white p-4 rounded-2xl shadow-sm">
                      <div className="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-600 flex items-center justify-center shrink-0">
                        {item.icon}
                      </div>
                      <span className="font-bold text-sm text-emerald-900">{item.text}</span>
                    </div>
                  ))}
                </div>
                <p className="mt-8 text-sm text-emerald-900/60 italic">
                  This approach helps learners understand the complete machine learning workflow, from data preparation to model evaluation.
                </p>
              </section>

              <section className="grid grid-cols-1 md:grid-cols-2 gap-12">
                <div className="space-y-6">
                  <h3 className="text-3xl font-bold">Our Vision</h3>
                  <p className="text-black/60 leading-relaxed">
                    The vision behind this platform is to create a <span className="font-bold text-black">virtual machine learning lab</span> where anyone can explore algorithms, test ideas, and understand how models behave in real time.
                  </p>
                  <div className="p-6 bg-stone-50 rounded-3xl border border-black/5">
                    <h4 className="text-xs font-bold uppercase text-black/40 mb-4">Questions you can answer:</h4>
                    <ul className="space-y-3 text-sm font-medium">
                      <li className="flex gap-3 text-black/70"><CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" /> What happens if I change the train-test split?</li>
                      <li className="flex gap-3 text-black/70"><CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" /> How does scaling affect the model?</li>
                      <li className="flex gap-3 text-black/70"><CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" /> Which algorithm works best for this dataset?</li>
                      <li className="flex gap-3 text-black/70"><CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" /> How do features influence prediction accuracy?</li>
                    </ul>
                  </div>
                </div>
                <div className="space-y-6">
                  <h3 className="text-3xl font-bold">Who This Is For</h3>
                  <p className="text-black/60 leading-relaxed">
                    ML Interactive Lab is designed for anyone curious about data science, regardless of their technical background.
                  </p>
                  <div className="space-y-4">
                    {[
                      "Students learning ML for the first time",
                      "Developers transitioning into AI",
                      "Instructors teaching ML concepts",
                      "Curious minds exploring data"
                    ].map((text, i) => (
                      <div key={i} className="flex items-center gap-3 p-4 border border-black/5 rounded-2xl hover:bg-stone-50 transition-colors">
                        <div className="w-2 h-2 rounded-full bg-emerald-600" />
                        <span className="text-sm font-bold">{text}</span>
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-black/40 italic">No complex setup is required — everything can be explored directly in the browser.</p>
                </div>
              </section>

              <section className="bg-black text-white p-12 rounded-[3rem] shadow-2xl">
                <div className="max-w-2xl">
                  <h3 className="text-3xl font-bold mb-6">Learning by Experimentation</h3>
                  <p className="text-white/60 leading-relaxed mb-8">
                    Machine learning is best understood through <span className="text-emerald-400 font-bold">experimentation</span>. We encourage users to try different combinations of datasets, algorithms, and configurations.
                  </p>
                  <div className="flex flex-wrap gap-3">
                    {['Datasets', 'Algorithms', 'Preprocessing', 'Configurations'].map(tag => (
                      <span key={tag} className="px-4 py-2 bg-white/10 rounded-full text-xs font-bold uppercase tracking-widest">{tag}</span>
                    ))}
                  </div>
                  <p className="mt-8 text-white/40 text-sm">
                    This hands-on approach transforms machine learning from something abstract into something clear, visual, and intuitive.
                  </p>
                </div>
              </section>

              <section className="text-center py-12">
                <h3 className="text-3xl font-bold mb-4">Our Goal</h3>
                <p className="text-black/60 max-w-2xl mx-auto mb-8">
                  Our goal is to make machine learning <span className="text-black font-bold">accessible, interactive, and practical for everyone</span>. We believe the best way to learn is by building, testing, and exploring yourself.
                </p>
                <button 
                  onClick={() => setShowAbout(false)}
                  className="px-12 py-4 bg-emerald-600 text-white rounded-2xl font-bold hover:bg-emerald-700 transition-all shadow-xl shadow-emerald-600/20"
                >
                  Start Experimenting Now
                </button>
              </section>
            </div>
          </motion.div>
        ) : showDocumentation ? (
          <DocumentationView onClose={() => setShowDocumentation(false)} />
        ) : showHistory ? (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="flex items-center gap-2 text-emerald-600 mb-8 cursor-pointer hover:underline" onClick={() => setShowHistory(false)}>
              <ChevronRight className="rotate-180 w-4 h-4" />
              <span className="text-sm font-medium">Back to Lab</span>
            </div>
            <ExperimentHistory 
              history={history} 
              onDelete={deleteExperiment} 
              onReRun={(item) => {
                reRunExperiment(item);
                setShowHistory(false);
              }}
              onCompare={(items) => {
                setComparisonModels(items.map(i => ({
                  id: i.id,
                  algoName: i.algo_name,
                  datasetId: i.dataset_id,
                  datasetName: i.dataset_name,
                  metrics: i.metrics,
                  hyperparams: i.hyperparams
                })));
                setIsComparisonMode(true);
                setShowHistory(false);
              }}
            />
          </motion.div>
        ) : (
          <>
            {step === 0 && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-24">
                {/* Hero Section */}
                <section className="text-center space-y-8 py-12">
                  <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.5 }}
                    className="inline-block px-4 py-1.5 bg-emerald-100 text-emerald-700 rounded-full text-[10px] font-bold uppercase tracking-widest mb-4"
                  >
                    Interactive Learning Platform
                  </motion.div>
                  <h1 className="text-6xl md:text-7xl font-bold tracking-tight max-w-4xl mx-auto leading-[1.1]">
                    Master Machine Learning through <span className="text-emerald-600">Experimentation.</span>
                  </h1>
                  <p className="text-xl text-black/50 max-w-2xl mx-auto leading-relaxed">
                    Select an algorithm, pick a dataset, and watch the model learn in real-time. No code required, just pure intuition.
                  </p>
                  <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
                    <button 
                      onClick={() => {
                        const el = document.getElementById('algo-selection');
                        el?.scrollIntoView({ behavior: 'smooth' });
                      }}
                      className="px-8 py-4 bg-black text-white rounded-2xl font-bold hover:bg-black/80 transition-all shadow-xl shadow-black/10 flex items-center justify-center gap-2"
                    >
                      Explore Algorithms <ArrowRight className="w-5 h-5" />
                    </button>
                    <button 
                      onClick={() => setShowAbout(true)}
                      className="px-8 py-4 bg-white border border-black/10 text-black rounded-2xl font-bold hover:bg-stone-50 transition-all"
                    >
                      How it Works
                    </button>
                  </div>
                </section>

                {/* Feature Grid */}
                <section className="grid grid-cols-1 md:grid-cols-3 gap-8">
                  {[
                    { icon: <Activity className="w-6 h-6" />, title: "Real-time Feedback", desc: "See how metrics change instantly as you tune hyperparameters." },
                    { icon: <BarChart2 className="w-6 h-6" />, title: "Visual Analytics", desc: "Interactive charts and decision boundaries to build intuition." },
                    { icon: <Layers className="w-6 h-6" />, title: "Full Pipeline", desc: "From data exploration to final evaluation and comparison." }
                  ].map((feature, i) => (
                    <div key={i} className="p-8 bg-white rounded-[2rem] border border-black/5 shadow-sm hover:shadow-md transition-all">
                      <div className="w-12 h-12 bg-emerald-50 text-emerald-600 rounded-2xl flex items-center justify-center mb-6">
                        {feature.icon}
                      </div>
                      <h3 className="text-lg font-bold mb-2">{feature.title}</h3>
                      <p className="text-sm text-black/50 leading-relaxed">{feature.desc}</p>
                    </div>
                  ))}
                </section>

                {/* Algorithm Selection */}
                <section id="algo-selection" className="space-y-12 pt-12">
                  <div className="text-center space-y-4">
                    <h2 className="text-4xl font-bold tracking-tight">Choose Your Learning Path</h2>
                    <p className="text-black/50 max-w-xl mx-auto">Select a learning paradigm to begin your journey. We've organized them by their primary function.</p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {['Supervised Learning', 'Unsupervised Learning'].map((category) => (
                      <div key={category} className={`space-y-6 ${category === 'Supervised Learning' ? 'md:col-span-2' : ''}`}>
                        <h3 className="text-xs uppercase tracking-widest font-bold text-black/40 border-b border-black/10 pb-2">{category}</h3>
                        
                        {category === 'Supervised Learning' ? (
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
                            <div className="space-y-4">
                              <h4 className="font-serif italic text-lg opacity-70">Regression</h4>
                              <div className="space-y-2">
                                {ALGORITHMS.filter(a => a.type === 'Regression').map(algo => (
                                  <AlgoButton key={algo.id} algo={algo} onClick={() => handleAlgoSelect(algo)} />
                                ))}
                              </div>
                            </div>
                            <div className="space-y-4">
                              <h4 className="font-serif italic text-lg opacity-70">Classification</h4>
                              <div className="space-y-2">
                                {ALGORITHMS.filter(a => a.type === 'Classification').map(algo => (
                                  <AlgoButton key={algo.id} algo={algo} onClick={() => handleAlgoSelect(algo)} />
                                ))}
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="space-y-4">
                            <h4 className="font-serif italic text-lg opacity-70">Clustering</h4>
                            <div className="space-y-2">
                              {ALGORITHMS.filter(a => a.type === 'Clustering').map(algo => (
                                <AlgoButton key={algo.id} algo={algo} onClick={() => handleAlgoSelect(algo)} />
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </section>
              </motion.div>
            )}

        {step === 1 && selectedAlgo && (
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
            <div className="flex items-center gap-2 text-emerald-600 mb-8 cursor-pointer hover:underline" onClick={() => setStep(0)}>
              <ChevronRight className="rotate-180 w-4 h-4" />
              <span className="text-sm font-medium">Back to Algorithms</span>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
              <div className="lg:col-span-1 space-y-6">
                <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm sticky top-24">
                  <div className="w-12 h-12 bg-emerald-100 text-emerald-600 rounded-2xl flex items-center justify-center mb-6">
                    <Microscope />
                  </div>
                  <h2 className="text-2xl font-bold mb-4">{selectedAlgo.name}</h2>
                  <p className="text-black/60 mb-6 leading-relaxed text-sm">{selectedAlgo.description}</p>
                  
                  <div className="space-y-4">
                    <div className="p-4 bg-emerald-50 rounded-xl">
                      <h4 className="text-[10px] font-bold uppercase text-emerald-600 mb-2 flex items-center gap-1">
                        <Zap className="w-3 h-3" /> How It Works
                      </h4>
                      <ul className="text-xs space-y-2 text-emerald-900/70">
                        {selectedAlgo.howItWorks.map((step, i) => (
                          <li key={i} className="flex gap-2">
                            <span className="font-bold text-emerald-600">•</span>
                            {step}
                          </li>
                        ))}
                      </ul>
                    </div>

                    {selectedAlgo.mathIntuition && (
                      <div className="p-4 bg-stone-100 rounded-xl">
                        <h4 className="text-[10px] font-bold uppercase text-black/40 mb-2 flex items-center gap-1">
                          <TrendingUp className="w-3 h-3" /> Math Intuition
                        </h4>
                        <p className="text-xs font-mono text-stone-600">{selectedAlgo.mathIntuition}</p>
                      </div>
                    )}

                    <div className="p-4 bg-blue-50 rounded-xl">
                      <h4 className="text-[10px] font-bold uppercase text-blue-600 mb-2 flex items-center gap-1">
                        <Target className="w-3 h-3" /> Real World Case
                      </h4>
                      <p className="text-xs text-blue-900/70 italic">{selectedAlgo.realWorldCase}</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="lg:col-span-2">
                <h3 className="text-2xl font-bold mb-8">Select a Dataset</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {FULL_DATASETS.filter(ds => ds.type === selectedAlgo.type.toLowerCase()).map(ds => (
                    <div 
                      key={ds.id} 
                      onClick={() => handleDatasetSelect(ds)}
                      className="group bg-white p-6 rounded-2xl border border-black/5 hover:border-emerald-600/50 hover:shadow-xl transition-all cursor-pointer"
                    >
                      <div className="flex justify-between items-start mb-4">
                        <div className="p-2 bg-emerald-50 text-emerald-600 rounded-lg">
                          <Database className="w-5 h-5" />
                        </div>
                        <ChevronRight className="w-5 h-5 opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                      <h4 className="font-bold text-lg mb-2">{ds.name}</h4>
                      <p className="text-sm text-black/50 line-clamp-2">{ds.description}</p>
                      <div className="mt-4 flex gap-2">
                        <span className="text-[10px] font-bold px-2 py-1 bg-black/5 rounded uppercase">{ds.data.length} samples</span>
                        <span className="text-[10px] font-bold px-2 py-1 bg-black/5 rounded uppercase">{ds.features.length} features</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {step === 2 && selectedDataset && (
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
            <div className="flex items-center gap-2 text-emerald-600 mb-8 cursor-pointer hover:underline" onClick={() => setStep(1)}>
              <ChevronRight className="rotate-180 w-4 h-4" />
              <span className="text-sm font-medium">Back to Datasets</span>
            </div>
            <div className="mb-12">
              <h2 className="text-4xl font-bold tracking-tight mb-4">Data Exploration</h2>
              <p className="text-black/60 max-w-2xl">Understand your data before training. Explore distributions, correlations, and statistical summaries.</p>
            </div>
            <DataExploration dataset={selectedDataset} />
            <div className="mt-12 flex justify-center">
              <button 
                onClick={() => setStep(3)}
                className="px-12 py-4 bg-emerald-600 text-white rounded-2xl font-bold hover:bg-emerald-700 transition-all shadow-lg shadow-emerald-600/20 flex items-center gap-2"
              >
                Continue to Pipeline <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </motion.div>
        )}

        {step === 3 && selectedDataset && selectedAlgo && (
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
             <div className="flex items-center gap-2 text-emerald-600 mb-8 cursor-pointer hover:underline" onClick={() => setStep(1)}>
              <ChevronRight className="rotate-180 w-4 h-4" />
              <span className="text-sm font-medium">Back to Datasets</span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
              <div className="lg:col-span-2 space-y-8">
                {/* Preprocessing Flow */}
                <section className="bg-white p-6 rounded-3xl border border-black/5 shadow-sm overflow-x-auto">
                  <div className="flex items-center justify-between min-w-[600px]">
                    <FlowStep icon={<Database className="w-4 h-4" />} label="Raw Data" active />
                    <ArrowRight className="w-4 h-4 text-black/20" />
                    <FlowStep icon={<Settings className="w-4 h-4" />} label="Feature Selection" active={features.length > 0} />
                    <ArrowRight className="w-4 h-4 text-black/20" />
                    <FlowStep icon={<Zap className="w-4 h-4" />} label="Scaling" active={scaling} />
                    <ArrowRight className="w-4 h-4 text-black/20" />
                    <FlowStep icon={<Layers className="w-4 h-4" />} label="Split" active />
                    <ArrowRight className="w-4 h-4 text-black/20" />
                    <FlowStep icon={<Cpu className="w-4 h-4" />} label="Training" active={isTraining} />
                  </div>
                </section>

                <section className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm">
                  <div className="flex items-center gap-3 mb-6">
                    <Settings className="text-emerald-600" />
                    <h3 className="text-xl font-bold">Preprocessing Configuration</h3>
                  </div>
                  
                  <div className="space-y-8">
                    {/* Warnings */}
                    <AnimatePresence>
                      {getWarnings().map(w => (
                        <motion.div 
                          key={w.id}
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          className={`p-4 rounded-xl flex items-start gap-3 ${w.type === 'warning' ? 'bg-amber-50 text-amber-900 border border-amber-100' : 'bg-blue-50 text-blue-900 border border-blue-100'}`}
                        >
                          {w.type === 'warning' ? <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" /> : <Info className="w-4 h-4 mt-0.5 shrink-0" />}
                          <p className="text-xs font-medium">{w.text}</p>
                        </motion.div>
                      ))}
                    </AnimatePresence>

                    <div>
                      <label className="text-xs font-bold uppercase text-black/40 mb-4 block">Feature Selection</label>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {selectedDataset.features.map(f => (
                          <label key={f} className="flex flex-col p-4 rounded-xl border border-black/5 cursor-pointer hover:bg-emerald-50 transition-colors group">
                            <div className="flex items-center gap-3 mb-2">
                              <input 
                                type="checkbox" 
                                checked={features.includes(f)} 
                                onChange={(e) => {
                                  if (e.target.checked) setFeatures([...features, f]);
                                  else setFeatures(features.filter(x => x !== f));
                                }}
                                className="w-4 h-4 accent-emerald-600"
                              />
                              <span className="text-sm font-bold">{f}</span>
                            </div>
                            {selectedDataset.columnDescriptions?.[f] && (
                              <p className="text-[10px] text-black/50 leading-tight pl-7">{selectedDataset.columnDescriptions[f]}</p>
                            )}
                          </label>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      <div>
                        <label className="text-xs font-bold uppercase text-black/40 mb-4 block">Feature Scaling</label>
                        <button 
                          onClick={() => setScaling(!scaling)}
                          className={`w-full p-4 rounded-xl border transition-all flex items-center justify-between ${scaling ? 'bg-emerald-600 text-white border-emerald-600' : 'bg-white text-black/60 border-black/5 hover:border-emerald-600/30'}`}
                        >
                          <span className="text-sm font-bold">Standardize Features</span>
                          {scaling ? <CheckCircle2 className="w-5 h-5" /> : <div className="w-5 h-5 rounded-full border-2 border-black/10" />}
                        </button>
                        <p className="text-[10px] text-black/40 mt-2">Scales features to have 0 mean and 1 variance.</p>
                      </div>

                      <div>
                        <label className="text-xs font-bold uppercase text-black/40 mb-4 block">Validation Strategy</label>
                        <div className="flex gap-4 mb-4">
                          <button 
                            onClick={() => setIsCrossValidation(false)}
                            className={`flex-1 py-3 rounded-xl border font-bold text-xs uppercase transition-all ${!isCrossValidation ? 'bg-emerald-600 text-white border-emerald-600' : 'bg-white text-black/40 border-black/5 hover:border-emerald-600/30'}`}
                          >
                            Train/Test Split
                          </button>
                          <button 
                            onClick={() => setIsCrossValidation(true)}
                            className={`flex-1 py-3 rounded-xl border font-bold text-xs uppercase transition-all ${isCrossValidation ? 'bg-emerald-600 text-white border-emerald-600' : 'bg-white text-black/40 border-black/5 hover:border-emerald-600/30'}`}
                          >
                            K-Fold CV
                          </button>
                        </div>
                        
                        {!isCrossValidation ? (
                          <>
                            <div className="flex justify-between mb-2">
                              <span className="text-xs font-bold text-black/60">Split Ratio ({Math.round(splitRatio * 100)}% Train)</span>
                            </div>
                            <input 
                              type="range" 
                              min="0.5" 
                              max="0.9" 
                              step="0.05" 
                              value={splitRatio} 
                              onChange={(e) => setSplitRatio(parseFloat(e.target.value))}
                              className="w-full h-2 bg-emerald-100 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                            />
                            <div className="flex justify-between mt-2 text-[10px] font-bold text-black/30">
                              <span>{Math.floor(selectedDataset.data.length * 0.5)} train</span>
                              <span>{Math.floor(selectedDataset.data.length * 0.9)} train</span>
                            </div>
                          </>
                        ) : (
                          <>
                            <div className="flex justify-between mb-2">
                              <span className="text-xs font-bold text-black/60">K-Folds ({kFolds})</span>
                            </div>
                            <input 
                              type="range" 
                              min="2" 
                              max="10" 
                              step="1" 
                              value={kFolds} 
                              onChange={(e) => setKFolds(parseInt(e.target.value))}
                              className="w-full h-2 bg-emerald-100 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                            />
                            <p className="text-[10px] text-black/40 mt-2">The dataset will be split into {kFolds} equal parts. The model will be trained {kFolds} times, each time using a different part as the test set.</p>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Hyperparameters */}
                    {selectedAlgo.hyperparameters && (
                      <div className="pt-6 border-t border-black/5">
                        <label className="text-xs font-bold uppercase text-black/40 mb-6 block">Hyperparameter Tuning</label>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                          {selectedAlgo.hyperparameters.map(h => (
                            <div key={h.id}>
                              <div className="flex justify-between mb-2">
                                <span className="text-sm font-bold">{h.name}</span>
                                <span className="text-sm font-mono text-emerald-600 font-bold">{hyperparams[h.id]}</span>
                              </div>
                              <input 
                                type="range" 
                                min={h.min} 
                                max={h.max} 
                                step={h.step} 
                                value={hyperparams[h.id] || h.default} 
                                onChange={(e) => setHyperparams({...hyperparams, [h.id]: parseFloat(e.target.value)})}
                                className="w-full h-2 bg-emerald-100 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                              />
                              <p className="text-[10px] text-black/40 mt-2">{h.description}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </section>

                <button 
                  onClick={handleTrain}
                  disabled={features.length === 0 || isTraining}
                  className="w-full py-4 bg-emerald-600 text-white rounded-2xl font-bold text-lg shadow-lg shadow-emerald-600/20 hover:bg-emerald-700 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isTraining ? (
                    <>
                      <RefreshCw className="w-5 h-5 animate-spin" />
                      Training Model...
                    </>
                  ) : (
                    <>
                      <Play fill="currentColor" className="w-5 h-5" />
                      Train Model
                    </>
                  )}
                </button>
              </div>

              <div className="lg:col-span-1 space-y-8">
                <div className="bg-white p-6 rounded-3xl border border-black/5 shadow-sm">
                  <h4 className="text-xs font-bold uppercase text-black/40 mb-4">Dataset Overview</h4>
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <div className="p-4 bg-stone-50 rounded-xl">
                      <span className="text-[10px] font-bold text-black/30 block uppercase">Samples</span>
                      <span className="text-xl font-bold">{selectedDataset.data.length}</span>
                    </div>
                    <div className="p-4 bg-stone-50 rounded-xl">
                      <span className="text-[10px] font-bold text-black/30 block uppercase">Selected Features</span>
                      <span className="text-xl font-bold">{features.length}</span>
                    </div>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-black/5">
                          {features.map(f => <th key={f} className="text-left py-2 font-bold pr-4">{f}</th>)}
                          {selectedDataset.target && <th className="text-left py-2 font-bold text-emerald-600">{selectedDataset.target}</th>}
                        </tr>
                      </thead>
                      <tbody>
                        {selectedDataset.data.slice(0, 5).map((row, i) => (
                          <tr key={i} className="border-b border-black/5 last:border-0">
                            {features.map(f => <td key={f} className="py-2 opacity-60 pr-4">{row[f]}</td>)}
                            {selectedDataset.target && <td key="target" className="py-2 font-medium">{row[selectedDataset.target]}</td>}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {features.length === 0 && (
                      <div className="py-8 text-center text-black/20 italic">
                        Select features to see a preview
                      </div>
                    )}
                  </div>
                  <p className="text-[10px] text-black/30 mt-4 text-center italic">Shape: ({selectedDataset.data.length}, {features.length + (selectedDataset.target ? 1 : 0)})</p>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {step === 4 && trainedModel && (
          <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}>
            <div className="flex justify-between items-center mb-12">
              <div>
                <h2 className="text-3xl font-bold tracking-tight">Model Evaluation</h2>
                <p className="text-black/50">{selectedAlgo?.name} trained on {selectedDataset?.name}</p>
              </div>
              <div className="flex gap-4">
                <button onClick={() => setStep(3)} className="flex items-center gap-2 px-6 py-3 bg-white border border-black/5 rounded-xl font-bold text-sm hover:bg-emerald-50 transition-colors">
                  <RefreshCw className="w-4 h-4" />
                  Back to Pipeline
                </button>
                <button onClick={reset} className="flex items-center gap-2 px-6 py-3 bg-black text-white rounded-xl font-bold text-sm hover:bg-black/80 transition-colors">
                  <RotateCcw className="w-4 h-4" />
                  Reset Lab
                </button>
              </div>
            </div>

            {/* Insights Section */}
            <div className="mb-12 bg-emerald-600 p-8 rounded-[2rem] text-white shadow-xl shadow-emerald-600/10">
              <div className="flex items-center gap-3 mb-4">
                <Zap className="w-6 h-6" />
                <h3 className="text-xl font-bold">Model Performance Insights</h3>
              </div>
              <p className="text-emerald-50 leading-relaxed opacity-90">{getInsights()}</p>
            </div>

            {/* Metrics Overview */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
              <div className="lg:col-span-2 space-y-12">
                {/* Section 1: Cross-Validation Performance */}
                {cvResults && (
                  <div className="space-y-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-600">
                        <Layers className="w-4 h-4" />
                      </div>
                      <div>
                        <h3 className="text-lg font-bold">Cross-Validation Performance (Training Validation)</h3>
                        <p className="text-xs text-black/40">This section shows the average performance of the model across multiple folds of the training data to estimate generalization.</p>
                      </div>
                    </div>
                    <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm">
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                        <MetricCard label="Avg Accuracy" value={`${(cvResults.avgMetrics.accuracy * 100).toFixed(1)}%`} color="emerald" />
                        <MetricCard label="Avg Precision" value={`${(cvResults.avgMetrics.precision * 100).toFixed(1)}%`} color="emerald" />
                        <MetricCard label="Avg Recall" value={`${(cvResults.avgMetrics.recall * 100).toFixed(1)}%`} color="emerald" />
                        <MetricCard label="Avg F1 Score" value={`${(cvResults.avgMetrics.f1 * 100).toFixed(1)}%`} color="emerald" />
                      </div>
                    </div>
                  </div>
                )}

                {/* Section 2: Test Set Performance */}
                {evaluation && (
                  <div className="space-y-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
                        <Target className="w-4 h-4" />
                      </div>
                      <div>
                        <h3 className="text-lg font-bold">Test Set Performance (Final Evaluation)</h3>
                        <p className="text-xs text-black/40">This section shows the final performance of the model on the unseen test dataset.</p>
                      </div>
                    </div>
                    <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm">
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                        {evaluation.accuracy !== undefined && (
                          <MetricCard label="Accuracy" value={`${(evaluation.accuracy * 100).toFixed(1)}%`} color="emerald" />
                        )}
                        {evaluation.precision !== undefined && (
                          <MetricCard label="Precision" value={`${(evaluation.precision * 100).toFixed(1)}%`} color="emerald" />
                        )}
                        {evaluation.recall !== undefined && (
                          <MetricCard label="Recall" value={`${(evaluation.recall * 100).toFixed(1)}%`} color="emerald" />
                        )}
                        {evaluation.f1 !== undefined && (
                          <MetricCard label="F1 Score" value={`${(evaluation.f1 * 100).toFixed(1)}%`} color="emerald" />
                        )}
                        {evaluation.r2 !== undefined && (
                          <MetricCard label="R² Score" value={evaluation.r2.toFixed(3)} color="emerald" />
                        )}
                        {evaluation.mse !== undefined && (
                          <MetricCard label="MSE" value={evaluation.mse > 1000000000 ? evaluation.mse.toExponential(2) : evaluation.mse.toFixed(2)} color="stone" />
                        )}
                        {evaluation.mae !== undefined && (
                          <MetricCard label="MAE" value={evaluation.mae > 1000000000 ? evaluation.mae.toExponential(2) : evaluation.mae.toFixed(2)} color="stone" />
                        )}
                        {evaluation.inertia !== undefined && (
                          <MetricCard label="Inertia" value={evaluation.inertia.toLocaleString(undefined, { maximumFractionDigits: 0 })} color="stone" />
                        )}
                      </div>

                      {/* R2 Warning */}
                      {evaluation.r2 !== undefined && evaluation.r2 < 0 && (
                        <div className="mt-6 p-4 bg-amber-50 border border-amber-100 rounded-2xl flex items-start gap-3">
                          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
                          <div className="text-xs text-amber-900">
                            <p className="font-bold mb-1">⚠ Model is performing worse than predicting the average value.</p>
                            <p className="opacity-80">An R² score below zero means your model's predictions are worse than just guessing the mean of the target. Try changing features, applying scaling, or using a different algorithm.</p>
                          </div>
                        </div>
                      )}

                      {/* Comparison Indicator */}
                      {cvResults && evaluation.accuracy !== undefined && (
                        <div className="mt-8 pt-8 border-t border-black/5">
                          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                            <div className="flex items-center gap-4">
                              <div className={`p-3 rounded-2xl ${Math.abs(cvResults.avgMetrics.accuracy - evaluation.accuracy) > 0.1 ? 'bg-amber-50 text-amber-600' : 'bg-emerald-50 text-emerald-600'}`}>
                                {Math.abs(cvResults.avgMetrics.accuracy - evaluation.accuracy) > 0.1 ? <AlertTriangle className="w-5 h-5" /> : <CheckCircle2 className="w-5 h-5" />}
                              </div>
                              <div>
                                <span className="text-[10px] font-bold uppercase text-black/40 block">Performance Gap (CV vs Test)</span>
                                <span className="text-sm font-bold">
                                  {Math.abs(cvResults.avgMetrics.accuracy - evaluation.accuracy) > 0.1 
                                    ? `Large Gap Detected: ${((cvResults.avgMetrics.accuracy - evaluation.accuracy) * 100).toFixed(1)}%`
                                    : `Healthy Generalization: ${((cvResults.avgMetrics.accuracy - evaluation.accuracy) * 100).toFixed(1)}%`
                                  }
                                </span>
                              </div>
                            </div>
                            {Math.abs(cvResults.avgMetrics.accuracy - evaluation.accuracy) > 0.1 && (
                              <div className="px-4 py-2 bg-amber-50 border border-amber-100 rounded-xl text-amber-700 text-[10px] font-bold uppercase">
                                Overfitting Warning
                              </div>
                            )}
                          </div>
                          <p className="text-[10px] text-black/40 mt-4">
                            A large gap between Cross-Validation and Test accuracy often indicates that the model has "memorized" the training data but fails to generalize to new, unseen data.
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* K-Means Animation */}
                {selectedAlgo?.id === 'kmeans' && kmeansHistory.length > 0 && (
                  <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm">
                    <div className="flex items-center justify-between mb-6">
                      <h4 className="text-xs font-bold uppercase text-black/40">K-Means Iterative Animation</h4>
                      <div className="flex items-center gap-4">
                        <span className="text-xs font-bold text-black/40">Iteration: {currentKMeansStep} / {kmeansHistory.length - 1}</span>
                        <div className="flex gap-2">
                          <button 
                            onClick={() => setCurrentKMeansStep(Math.max(0, currentKMeansStep - 1))}
                            className="p-2 bg-black/5 rounded-lg hover:bg-black/10 transition-colors"
                          >
                            <ChevronRight className="rotate-180 w-4 h-4" />
                          </button>
                          <button 
                            onClick={() => setCurrentKMeansStep(Math.min(kmeansHistory.length - 1, currentKMeansStep + 1))}
                            className="p-2 bg-black/5 rounded-lg hover:bg-black/10 transition-colors"
                          >
                            <ChevronRight className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                    <div className="h-[400px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 80 }}>
                          <XAxis type="number" dataKey="x" name={features[0]} />
                          <YAxis 
                            type="number" 
                            dataKey="y" 
                            name={features[1] || 'Y'} 
                            width={80}
                            tickFormatter={(value) => value.toLocaleString('en-US', { notation: 'compact', compactDisplay: 'short' })}
                          />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                          <Scatter name="Data" data={selectedDataset.data.map((d, i) => ({
                            x: parseFloat(d[features[0]]),
                            y: features[1] ? parseFloat(d[features[1]]) : 0,
                            cluster: kmeansHistory[currentKMeansStep].clusters[i]
                          }))}>
                            {selectedDataset.data.map((_, index) => {
                              const colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'];
                              const cluster = kmeansHistory[currentKMeansStep].clusters[index];
                              return <Cell key={`cell-${index}`} fill={cluster !== undefined ? colors[cluster % colors.length] : '#ccc'} opacity={0.6} />;
                            })}
                          </Scatter>
                          <Scatter name="Centroids" data={kmeansHistory[currentKMeansStep].centroids.map((c, i) => ({
                            x: c[0],
                            y: c[1] || 0,
                            id: i
                          }))}>
                            {kmeansHistory[currentKMeansStep].centroids.map((_, index) => (
                              <Cell key={`centroid-${index}`} fill="#000" stroke="#fff" strokeWidth={2} />
                            ))}
                          </Scatter>
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {/* ROC Curve */}
                {rocData && (
                  <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm h-[400px]">
                    <h4 className="text-xs font-bold uppercase text-black/40 mb-6">ROC Curve (Binary Classification)</h4>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={rocData} margin={{ top: 10, right: 30, left: 40, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                        <XAxis dataKey="fpr" label={{ value: 'False Positive Rate', position: 'insideBottomRight', offset: -10 }} />
                        <YAxis 
                          width={50}
                          label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', offset: 10 }} 
                        />
                        <Tooltip />
                        <Area type="monotone" dataKey="tpr" stroke="#10b981" fill="#10b981" fillOpacity={0.1} strokeWidth={3} />
                        <Line type="monotone" data={[{fpr: 0, tpr: 0}, {fpr: 1, tpr: 1}]} dataKey="tpr" stroke="#ccc" strokeDasharray="5 5" dot={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Learning Curve */}
                {learningCurveData && (
                  <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm h-[400px]">
                    <h4 className="text-xs font-bold uppercase text-black/40 mb-6">Learning Curve (Error vs Training Samples)</h4>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={learningCurveData} margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                        <XAxis dataKey="size" label={{ value: 'Training Samples', position: 'insideBottomRight', offset: -10 }} />
                        <YAxis 
                          width={80}
                          tickFormatter={(value) => value.toLocaleString('en-US', { notation: 'compact', compactDisplay: 'short' })}
                          label={{ value: trainedModel?.type === 'classification' ? 'Error (1 - Accuracy)' : 'Mean Squared Error', angle: -90, position: 'insideLeft', offset: 10 }} 
                        />
                        <Tooltip />
                        <Legend verticalAlign="top" height={36}/>
                        <Line type="monotone" dataKey="trainError" name="Training Error" stroke="#10b981" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="testError" name="Validation Error" stroke="#ef4444" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Bias-Variance Tradeoff */}
                {biasVarianceData && (
                  <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm h-[400px]">
                    <h4 className="text-xs font-bold uppercase text-black/40 mb-6">Bias-Variance Tradeoff (Complexity vs Error)</h4>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={biasVarianceData} margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                        <XAxis dataKey="complexity" label={{ value: `Complexity (${selectedAlgo?.complexityParameter})`, position: 'insideBottomRight', offset: -10 }} />
                        <YAxis 
                          width={80}
                          tickFormatter={(value) => value.toLocaleString('en-US', { notation: 'compact', compactDisplay: 'short' })}
                          label={{ value: 'Error', angle: -90, position: 'insideLeft', offset: 10 }} 
                        />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="trainError" name="Bias (Train Error)" stroke="#3b82f6" strokeWidth={2} />
                        <Line type="monotone" dataKey="testError" name="Variance (Test Error)" stroke="#f59e0b" strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Decision Tree Structure */}
                {selectedAlgo?.id === 'decision-tree' && treeData && (
                  <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm overflow-auto">
                    <h4 className="text-xs font-bold uppercase text-black/40 mb-6">Decision Tree Structure</h4>
                    <div className="font-mono text-[10px] whitespace-pre p-4 bg-stone-50 rounded-xl">
                      {JSON.stringify(treeData, null, 2)}
                    </div>
                  </div>
                )}

                {/* Clustering Scatter Plot */}
                {selectedAlgo?.type === 'Clustering' && trainedModel && (
                  <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm h-[400px]">
                    <h4 className="text-xs font-bold uppercase text-black/40 mb-6">Cluster Visualization (First 2 Features)</h4>
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 80 }}>
                        <XAxis type="number" dataKey="x" name={features[0]} />
                        <YAxis 
                          type="number" 
                          dataKey="y" 
                          name={features[1] || 'Y'} 
                          width={80}
                          tickFormatter={(value) => value.toLocaleString('en-US', { notation: 'compact', compactDisplay: 'short' })}
                        />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Scatter name="Clusters" data={selectedDataset.data.map((d, i) => {
                          const x = parseFloat(d[features[0]]);
                          const y = features[1] ? parseFloat(d[features[1]]) : 0;
                          const input = features.map(f => parseFloat(d[f]));
                          const cluster = predict(trainedModel, input);
                          return { x, y, cluster };
                        })}>
                          {selectedDataset.data.map((_, index) => {
                            const colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'];
                            const d = selectedDataset.data[index];
                            const input = features.map(f => parseFloat(d[f]));
                            const cluster = predict(trainedModel, input);
                            return <Cell key={`cell-${index}`} fill={colors[cluster % colors.length]} />;
                          })}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Logistic Regression S-Curve */}
                {selectedAlgo?.id === 'logistic-regression' && trainedModel?.model?.weights && (
                  <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm h-[400px]">
                    <h4 className="text-xs font-bold uppercase text-black/40 mb-6">Logistic S-Curve (Probability vs Logit)</h4>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart margin={{ top: 20, right: 20, bottom: 20, left: 60 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                        <XAxis type="number" dataKey="logit" name="Logit (Linear Combination)" domain={['auto', 'auto']} />
                        <YAxis 
                          type="number" 
                          domain={[0, 1]} 
                          width={60}
                          label={{ value: 'Probability', angle: -90, position: 'insideLeft', offset: 10 }} 
                        />
                        <Tooltip />
                        <Line 
                          type="monotone" 
                          data={Array.from({ length: 100 }, (_, i) => {
                            const logit = -10 + (i * 20) / 99;
                            const prob = 1 / (1 + Math.exp(-logit));
                            return { logit, prob };
                          })} 
                          dataKey="prob" 
                          stroke="#10b981" 
                          strokeWidth={3} 
                          dot={false} 
                        />
                        <Scatter 
                          data={selectedDataset.data.map(d => {
                            const input = features.map(f => parseFloat(d[f]));
                            let processedInput = [...input];
                            if (trainedModel.scalingParams) {
                              const { means, stds } = trainedModel.scalingParams;
                              processedInput = processedInput.map((val, i) => (val - means[i]) / stds[i]);
                            }
                            const weights = trainedModel.model.weights;
                            let logit = 0;
                            for(let i=0; i<processedInput.length; i++) {
                              logit += processedInput[i] * weights[i];
                            }
                            const prob = 1 / (1 + Math.exp(-logit));
                            return { logit, prob, actual: d[target] };
                          })}
                        >
                          {selectedDataset.data.map((d, index) => {
                            const actual = d[target];
                            const isPositive = actual === trainedModel.reverseTargetMap[1] || actual === 1 || actual === '1' || actual === 'Yes';
                            return <Cell key={`cell-data-${index}`} fill={isPositive ? '#065f46' : '#991b1b'} opacity={0.5} />;
                          })}
                        </Scatter>
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Overfitting Check */}
                {evaluation && trainEvaluation && evaluation.accuracy !== undefined && (
                  <div className="bg-white p-6 rounded-3xl border border-black/5 shadow-sm">
                    <div className="flex items-center justify-between mb-6">
                      <h4 className="text-xs font-bold uppercase text-black/40">Train vs Test Performance</h4>
                      {trainEvaluation.accuracy - evaluation.accuracy > 0.15 && (
                        <div className="flex items-center gap-2 text-amber-600 bg-amber-50 px-3 py-1 rounded-full text-[10px] font-bold uppercase border border-amber-100">
                          <AlertTriangle className="w-3 h-3" /> Overfitting Detected
                        </div>
                      )}
                    </div>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={[
                          { name: 'Training', value: trainEvaluation.accuracy || trainEvaluation.r2 },
                          { name: 'Testing', value: evaluation.accuracy || evaluation.r2 }
                        ]}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                          <XAxis dataKey="name" />
                          <YAxis domain={[0, 1]} />
                          <Tooltip />
                          <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                            { [0, 1].map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={index === 0 ? '#10b981' : '#3b82f6'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <p className="text-[10px] text-black/40 mt-4 text-center">
                      {trainEvaluation.accuracy - evaluation.accuracy > 0.15 
                        ? "Your model performs much better on training data than unseen test data. Try reducing complexity or adding regularization."
                        : "Your model shows good generalization between training and testing data."}
                    </p>
                  </div>
                )}

                {/* Improvement Suggestions */}
                {evaluation && (evaluation.accuracy < 0.7 || evaluation.r2 < 0.7) && (
                  <div className="bg-emerald-50 p-6 rounded-3xl border border-emerald-100">
                    <h4 className="text-xs font-bold uppercase text-emerald-600 mb-4 flex items-center gap-2">
                      <Zap className="w-4 h-4" /> Suggestions to Improve
                    </h4>
                    <ul className="text-sm space-y-2 text-emerald-900/70">
                      <li className="flex gap-2">• Try selecting different features that might have more predictive power.</li>
                      <li className="flex gap-2">• Adjust hyperparameters (e.g., increase K in KNN or depth in Decision Trees).</li>
                      <li className="flex gap-2">• Ensure your data is properly scaled if using distance-based models.</li>
                      <li className="flex gap-2">• Consider if the chosen algorithm is suitable for this specific dataset.</li>
                    </ul>
                  </div>
                )}
              </div>
              
              <div className="lg:col-span-1 space-y-8">
                <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm">
                  <div className="flex items-center gap-2 mb-6">
                    <Play className="text-emerald-600 w-5 h-5" />
                    <h3 className="text-xl font-bold">Test Prediction</h3>
                  </div>
                  <div className="space-y-4">
                    {features.map(f => (
                      <div key={f}>
                        <label className="text-[10px] font-bold uppercase text-black/40 mb-1 block">{f}</label>
                        <input 
                          type="number" 
                          value={testInput[f] || ''} 
                          onChange={(e) => setTestInput({...testInput, [f]: e.target.value})}
                          className="w-full p-3 bg-[#F5F5F0] rounded-xl border border-transparent focus:border-emerald-600 focus:bg-white transition-all outline-none text-sm"
                          placeholder="Enter value..."
                        />
                      </div>
                    ))}
                    <button 
                      onClick={handlePredict}
                      className="w-full py-3 bg-emerald-600 text-white rounded-xl font-bold hover:bg-emerald-700 transition-colors mt-4"
                    >
                      Predict
                    </button>
                    {prediction !== null && (
                      <motion.div 
                        initial={{ opacity: 0, y: 10 }} 
                        animate={{ opacity: 1, y: 0 }}
                        className="mt-6 p-4 bg-emerald-50 border border-emerald-100 rounded-2xl text-center"
                      >
                        <span className="text-xs font-bold uppercase text-emerald-600 block mb-1">Prediction Output</span>
                        <span className="text-2xl font-bold text-emerald-900">{prediction}</span>
                      </motion.div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Visualizations */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
              {/* Feature Importance */}
              {featureImportance && (
                <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm h-[400px]">
                  <h4 className="text-xs font-bold uppercase text-black/40 mb-6">Feature Importance</h4>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart layout="vertical" data={featureImportance} margin={{ left: 40, right: 40 }}>
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f0f0f0" />
                      <XAxis type="number" hide domain={[0, 'auto']} />
                      <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 10 }} />
                      <Tooltip formatter={(value: number) => [value.toFixed(4), 'Importance']} />
                      <Bar 
                        dataKey="value" 
                        fill="#10b981" 
                        radius={[0, 4, 4, 0]} 
                        label={{ position: 'right', fontSize: 10, fill: '#666', formatter: (v: number) => v > 0 ? v.toFixed(2) : '' }}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Decision Boundary */}
              {decisionBoundary && (
                <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm h-[400px] relative">
                  <h4 className="text-xs font-bold uppercase text-black/40 mb-6">Decision Boundary (2D)</h4>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }} onClick={(e) => {
                      if (e && e.activePayload && e.activePayload[0]) {
                        const point = e.activePayload[0].payload;
                        setSelectedPoint(point);
                        if (selectedAlgo?.id === 'knn' && trainedModel) {
                          const neighbors = trainedModel.model.getNeighbors([point.x, point.y], hyperparams.k || 3);
                          setKNNNeighbors(neighbors);
                        }
                      }
                    }}>
                      <XAxis type="number" dataKey="x" name={features[0]} domain={decisionBoundary.xRange} />
                      <YAxis type="number" dataKey="y" name={features[1]} domain={decisionBoundary.yRange} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Scatter name="Boundary" data={decisionBoundary.grid}>
                        {decisionBoundary.grid.map((entry: any, index: number) => {
                          let fill = '#f59e0b';
                          let opacity = 0.1;
                          if (evaluation.classes) {
                            fill = entry.pred === evaluation.classes[0] ? '#10b981' : entry.pred === evaluation.classes[1] ? '#3b82f6' : '#f59e0b';
                          } else {
                            const preds = decisionBoundary.grid.map((g: any) => g.pred);
                            const min = preds.length > 0 ? Math.min(...preds) : 0;
                            const max = preds.length > 0 ? Math.max(...preds) : 1;
                            const range = max - min;
                            const normalized = range === 0 ? 0.5 : (entry.pred - min) / range;
                            fill = `rgba(16, 185, 129, ${normalized})`;
                          }

                          // Show SVM Margin
                          if (selectedAlgo?.id === 'svm' && entry.margin !== undefined) {
                            const absMargin = Math.abs(entry.margin);
                            if (absMargin < 0.1) opacity = 0.4; // Near boundary
                            else if (absMargin < 0.5) opacity = 0.2;
                          }

                          return <Cell key={`cell-${index}`} fill={fill} opacity={opacity} />;
                        })}
                      </Scatter>
                      <Scatter name="Data" data={selectedDataset.data.map(d => ({ x: parseFloat(d[features[0]]), y: parseFloat(d[features[1]]), pred: d[target] }))}>
                        {selectedDataset.data.map((entry: any, index: number) => {
                          let fill = '#92400e';
                          if (evaluation.classes) {
                            fill = entry[target] === evaluation.classes[0] ? '#065f46' : entry[target] === evaluation.classes[1] ? '#1e40af' : '#92400e';
                          } else {
                            const targetValues = selectedDataset.data.map(d => parseFloat(d[target]));
                            const min = targetValues.length > 0 ? Math.min(...targetValues) : 0;
                            const max = targetValues.length > 0 ? Math.max(...targetValues) : 1;
                            const range = max - min;
                            const normalized = range === 0 ? 0.5 : (parseFloat(entry[target]) - min) / range;
                            fill = `rgba(6, 95, 70, ${normalized})`;
                          }
                          return <Cell key={`cell-data-${index}`} fill={fill} />;
                        })}
                      </Scatter>

                      {/* Support Vectors */}
                      {selectedAlgo?.id === 'svm' && trainedModel?.model?.supportVectors && (
                        <Scatter name="Support Vectors" data={trainedModel.model.supportVectors.map((sv: any) => ({
                          x: sv[0],
                          y: sv[1] || 0
                        }))}>
                          {trainedModel.model.supportVectors.map((_: any, i: number) => (
                            <Cell key={`sv-${i}`} stroke="#000" strokeWidth={2} fill="none" r={6} />
                          ))}
                        </Scatter>
                      )}

                      {/* KNN Neighbors */}
                      {selectedAlgo?.id === 'knn' && knnNeighbors.length > 0 && selectedPoint && (
                        <Scatter name="Neighbors" data={knnNeighbors.map((n: any) => ({
                          x: n[0][0],
                          y: n[0][1] || 0
                        }))}>
                          {knnNeighbors.map((_: any, i: number) => (
                            <Cell key={`neighbor-${i}`} stroke="#f59e0b" strokeWidth={2} fill="none" r={8} />
                          ))}
                        </Scatter>
                      )}
                    </ScatterChart>
                  </ResponsiveContainer>
                  {selectedAlgo?.id === 'knn' && (
                    <div className="absolute top-4 right-4 bg-white/80 backdrop-blur p-2 rounded-lg text-[10px] border border-black/5">
                      Click a point to see its {hyperparams.k || 3} nearest neighbors
                    </div>
                  )}
                </div>
              )}

              {/* Confusion Matrix */}
              {evaluation?.confusionMatrix && (
                <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm min-h-[400px] flex flex-col">
                  <div className="flex items-center justify-between mb-6">
                    <h4 className="text-xs font-bold uppercase text-black/40">Confusion Matrix</h4>
                    <div className="group relative">
                      <HelpCircle className="w-4 h-4 text-black/20 cursor-help" />
                      <div className="absolute right-0 top-6 w-64 p-4 bg-black text-white text-[10px] rounded-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 leading-relaxed">
                        <p className="font-bold mb-2">How to read:</p>
                        <p className="mb-1"><span className="text-emerald-400">Diagonal:</span> Correct predictions (True Positives/Negatives).</p>
                        <p><span className="text-amber-400">Off-diagonal:</span> Misclassifications (False Positives/Negatives).</p>
                      </div>
                    </div>
                  </div>
                  <div className="flex-1 flex items-center justify-center overflow-auto">
                    <div className="grid gap-1 w-full max-w-md mx-auto" style={{ gridTemplateColumns: `repeat(${evaluation.classes.length}, 1fr)` }}>
                    {evaluation.confusionMatrix.map((row: number[], i: number) => (
                      row.map((val: number, j: number) => (
                        <div 
                          key={`${i}-${j}`} 
                          className={`aspect-square flex items-center justify-center rounded-md text-sm font-bold transition-all hover:scale-105 cursor-default ${i === j ? 'bg-emerald-500 text-white' : 'bg-stone-100 text-stone-400'}`}
                          title={`Actual: ${evaluation.classes[i]}, Predicted: ${evaluation.classes[j]}`}
                        >
                          {val}
                        </div>
                      ))
                    ))}
                  </div>
                </div>
                <div className="mt-4 flex justify-between text-[10px] font-bold text-black/30 uppercase">
                    <span className="flex items-center gap-1">Actual <ArrowRight className="w-3 h-3 rotate-90" /></span>
                    <span className="flex items-center gap-1">Predicted <ArrowRight className="w-3 h-3" /></span>
                  </div>
                </div>
              )}
            </div>

            {/* Comparison Mode Toggle */}
            <div className="mb-16">
              <div className="flex items-center justify-between mb-8">
                <h3 className="text-2xl font-bold">Model Comparison</h3>
                <button 
                  onClick={() => setIsComparisonMode(!isComparisonMode)}
                  className={`px-6 py-2 rounded-full font-bold text-xs uppercase tracking-widest transition-all ${isComparisonMode ? 'bg-emerald-600 text-white' : 'bg-black/5 text-black/40 hover:bg-black/10'}`}
                >
                  {isComparisonMode ? 'Comparison Active' : 'Enable Comparison'}
                </button>
              </div>
              
              {isComparisonMode && (
                <div className="space-y-8">
                  <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-black/5 text-left">
                          <th className="py-4 font-bold text-black/40 uppercase text-[10px]">Algorithm</th>
                          <th className="py-4 font-bold text-black/40 uppercase text-[10px]">Accuracy / R²</th>
                          <th className="py-4 font-bold text-black/40 uppercase text-[10px]">Precision</th>
                          <th className="py-4 font-bold text-black/40 uppercase text-[10px]">Recall</th>
                          <th className="py-4 font-bold text-black/40 uppercase text-[10px]">F1 Score</th>
                          <th className="py-4 font-bold text-black/40 uppercase text-[10px]">Train Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparisonModels.filter(m => m.datasetId === selectedDataset.id).map((m, i) => {
                          const currentDatasetModels = comparisonModels.filter(mod => mod.datasetId === selectedDataset.id);
                          const accs = currentDatasetModels.map(mod => mod.metrics.accuracy || mod.metrics.r2);
                          const bestAcc = accs.length > 0 ? Math.max(...accs) : 0;
                          const isBest = (m.metrics.accuracy || m.metrics.r2) === bestAcc;
                          
                          return (
                            <tr key={m.id} className={`border-b border-black/5 last:border-0 ${isBest ? 'bg-emerald-50/50' : ''}`}>
                              <td className="py-4 font-bold flex items-center gap-2">
                                {m.algoName}
                                {isBest && <span className="text-[8px] bg-emerald-600 text-white px-1.5 py-0.5 rounded-full uppercase">Best</span>}
                              </td>
                              <td className="py-4 text-emerald-600 font-bold">{(m.metrics.accuracy !== undefined ? m.metrics.accuracy : m.metrics.r2).toFixed(3)}</td>
                              <td className="py-4">{m.metrics.precision?.toFixed(3) || '-'}</td>
                              <td className="py-4">{m.metrics.recall?.toFixed(3) || '-'}</td>
                              <td className="py-4">{m.metrics.f1?.toFixed(3) || '-'}</td>
                              <td className="py-4 font-mono text-[10px]">{m.metrics.trainingTime?.toFixed(1) || '-'}ms</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm h-80">
                    <h4 className="text-xs font-bold uppercase text-black/40 mb-6">Performance Comparison Chart</h4>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={comparisonModels.filter(m => m.datasetId === selectedDataset.id).map(m => ({
                        name: m.algoName,
                        accuracy: m.metrics.accuracy || m.metrics.r2
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                        <XAxis dataKey="name" />
                        <YAxis domain={[0, 1]} />
                        <Tooltip />
                        <Bar dataKey="accuracy" fill="#10b981" radius={[8, 8, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>

            {/* Next Steps */}
            {/* Algorithm Deep Dive */}
            {selectedAlgo?.educationalDetails && (
              <div className="mb-16">
                <div className="flex items-center gap-3 mb-8">
                  <div className="w-10 h-10 bg-emerald-100 text-emerald-600 rounded-xl flex items-center justify-center">
                    <Microscope className="w-5 h-5" />
                  </div>
                  <h3 className="text-2xl font-bold">Algorithm Deep Dive: {selectedAlgo.name}</h3>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm">
                    <h4 className="text-xs font-bold uppercase text-emerald-600 mb-4 flex items-center gap-2">
                      <Zap className="w-4 h-4" /> What it does & How it works
                    </h4>
                    <div className="space-y-4">
                      <p className="text-sm text-black/70 leading-relaxed">
                        {selectedAlgo.educationalDetails.whatItDoes}
                      </p>
                      <div className="p-4 bg-emerald-50/50 rounded-2xl border border-emerald-100">
                        <p className="text-sm text-emerald-900 leading-relaxed">
                          {selectedAlgo.educationalDetails.howItWorks}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-8">
                    <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm">
                      <h4 className="text-xs font-bold uppercase text-blue-600 mb-4 flex items-center gap-2">
                        <TrendingUp className="w-4 h-4" /> Mathematical Intuition
                      </h4>
                      <p className="text-sm text-black/70 leading-relaxed italic">
                        "{selectedAlgo.educationalDetails.mathIntuition}"
                      </p>
                    </div>

                    <div className="bg-white p-8 rounded-3xl border border-black/5 shadow-sm">
                      <h4 className="text-xs font-bold uppercase text-amber-600 mb-4 flex items-center gap-2">
                        <Target className="w-4 h-4" /> Real-World Applications
                      </h4>
                      <ul className="space-y-3">
                        {selectedAlgo.educationalDetails.realWorldExamples.map((example, i) => (
                          <li key={i} className="flex items-start gap-3 text-sm text-black/70">
                            <div className="w-5 h-5 bg-amber-100 text-amber-600 rounded-full flex-shrink-0 flex items-center justify-center text-[10px] font-bold">
                              {i + 1}
                            </div>
                            {example}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mt-16 p-12 bg-emerald-600 rounded-[3rem] text-white text-center shadow-2xl shadow-emerald-600/20"
            >
              <h3 className="text-3xl font-bold mb-4">Ready for more?</h3>
              <p className="text-emerald-50 mb-8 max-w-xl mx-auto opacity-80">You've successfully trained and evaluated your model. You can now try a different algorithm or use a new dataset to see how it performs.</p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <button 
                  onClick={reset}
                  className="px-8 py-4 bg-white text-emerald-600 rounded-2xl font-bold hover:bg-emerald-50 transition-all flex items-center justify-center gap-2"
                >
                  <RefreshCw className="w-5 h-5" />
                  Train Another Model
                </button>
                <button 
                  onClick={() => setStep(0)}
                  className="px-8 py-4 bg-emerald-700 text-white rounded-2xl font-bold hover:bg-emerald-800 transition-all"
                >
                  Switch Algorithm
                </button>
              </div>
            </motion.div>
            </motion.div>
            )}
          </>
        )}
      </main>

      <footer className="mt-24 border-t border-black/5 py-12 px-6">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="flex items-center gap-2 opacity-50">
            <div className="w-6 h-6 bg-black rounded flex items-center justify-center text-white text-[10px] font-bold">ML</div>
            <span className="text-xs font-medium">Interactive Lab © 2026</span>
          </div>
          <div className="flex flex-wrap justify-center md:justify-end gap-4 text-xs font-bold uppercase tracking-widest">
            <button 
              onClick={() => {
                setShowDocumentation(true);
                setShowAbout(false);
                setShowHistory(false);
              }}
              className="px-6 py-3 bg-black/5 hover:bg-emerald-600 hover:text-white rounded-xl transition-all flex items-center gap-2"
            >
              <BookOpen className="w-4 h-4" />
              Documentation
            </button>
            <a 
              href="https://github.com/python123-dev" 
              target="_blank" 
              rel="noopener noreferrer"
              className="px-6 py-3 bg-black/5 hover:bg-emerald-600 hover:text-white rounded-xl transition-all flex items-center gap-2"
            >
              <Github className="w-4 h-4" />
              Github
            </a>
            <button 
              onClick={() => {
                setShowAbout(true);
                setShowDocumentation(false);
                setShowHistory(false);
              }} 
              className="px-6 py-3 bg-black/5 hover:bg-emerald-600 hover:text-white rounded-xl transition-all flex items-center gap-2"
            >
              <Info className="w-4 h-4" />
              About
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}

const DocumentationView = ({ onClose }: { onClose: () => void }) => {
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-4xl mx-auto">
      <div className="flex items-center gap-2 text-emerald-600 mb-12 cursor-pointer hover:underline" onClick={onClose}>
        <ChevronRight className="rotate-180 w-4 h-4" />
        <span className="text-sm font-medium">Back to Lab</span>
      </div>
      
      <div className="space-y-16">
        <section>
          <h2 className="text-5xl font-bold tracking-tight mb-8">Documentation</h2>
          <p className="text-xl text-black/60 leading-relaxed">
            Welcome to the ML Interactive Lab documentation. This guide will help you understand how to use the platform to explore machine learning concepts.
          </p>
        </section>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="p-8 bg-white rounded-[2rem] border border-black/5 shadow-sm">
            <div className="w-12 h-12 rounded-2xl bg-emerald-100 text-emerald-600 flex items-center justify-center mb-6">
              <Layers className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold mb-4">1. Select Algorithm</h3>
            <p className="text-sm text-black/50 leading-relaxed">
              Choose from various machine learning algorithms like Linear Regression, Decision Trees, or Random Forests. Each algorithm has its own strengths and use cases.
            </p>
          </div>

          <div className="p-8 bg-white rounded-[2rem] border border-black/5 shadow-sm">
            <div className="w-12 h-12 rounded-2xl bg-blue-100 text-blue-600 flex items-center justify-center mb-6">
              <Database className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold mb-4">2. Choose Dataset</h3>
            <p className="text-sm text-black/50 leading-relaxed">
              Pick a dataset to train your model on. We provide classic datasets like Iris, Wine, and Breast Cancer, or you can use synthetic data for simpler experiments.
            </p>
          </div>

          <div className="p-8 bg-white rounded-[2rem] border border-black/5 shadow-sm">
            <div className="w-12 h-12 rounded-2xl bg-purple-100 text-purple-600 flex items-center justify-center mb-6">
              <Settings className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold mb-4">3. Configure Features</h3>
            <p className="text-sm text-black/50 leading-relaxed">
              Select which features to include in your model and choose preprocessing steps like scaling or cross-validation to improve performance.
            </p>
          </div>

          <div className="p-8 bg-white rounded-[2rem] border border-black/5 shadow-sm">
            <div className="w-12 h-12 rounded-2xl bg-orange-100 text-orange-600 flex items-center justify-center mb-6">
              <Activity className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold mb-4">4. Evaluate Results</h3>
            <p className="text-sm text-black/50 leading-relaxed">
              After training, analyze performance metrics like Accuracy, MSE, or R² score. Use the interactive visualizations to understand how your model makes predictions.
            </p>
          </div>
        </div>

        <section className="bg-stone-900 text-white p-12 rounded-[3rem]">
          <h3 className="text-2xl font-bold mb-6">Pro Tips</h3>
          <ul className="space-y-4">
            <li className="flex gap-4">
              <div className="w-6 h-6 rounded-full bg-emerald-500 flex items-center justify-center shrink-0 text-[10px] font-bold">1</div>
              <p className="text-white/60 text-sm">Use <strong>Scaling</strong> when working with algorithms that are sensitive to feature magnitude, like KNN or SVM.</p>
            </li>
            <li className="flex gap-4">
              <div className="w-6 h-6 rounded-full bg-emerald-500 flex items-center justify-center shrink-0 text-[10px] font-bold">2</div>
              <p className="text-white/60 text-sm">Enable <strong>Cross-Validation</strong> for a more reliable estimate of model performance on unseen data.</p>
            </li>
            <li className="flex gap-4">
              <div className="w-6 h-6 rounded-full bg-emerald-500 flex items-center justify-center shrink-0 text-[10px] font-bold">3</div>
              <p className="text-white/60 text-sm">Check the <strong>Feature Importance</strong> chart to see which variables are driving your model's decisions.</p>
            </li>
          </ul>
        </section>
      </div>
    </motion.div>
  );
};

function FlowStep({ icon, label, active }: { icon: React.ReactNode, label: string, active?: boolean }) {
  return (
    <div className={`flex flex-col items-center gap-2 transition-all ${active ? 'opacity-100 scale-110' : 'opacity-30'}`}>
      <div className={`w-10 h-10 rounded-full flex items-center justify-center ${active ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-600/20' : 'bg-black/5 text-black'}`}>
        {icon}
      </div>
      <span className="text-[10px] font-bold uppercase tracking-tighter">{label}</span>
    </div>
  );
}

const AlgoButton: React.FC<{ algo: AlgorithmInfo, onClick: () => void }> = ({ algo, onClick }) => {
  return (
    <button 
      onClick={onClick}
      className="w-full text-left p-4 rounded-xl border border-black/5 bg-white hover:border-emerald-600/50 hover:shadow-md transition-all group"
    >
      <div className="flex items-center justify-between">
        <div className="flex flex-col">
          <span className="font-bold text-sm">{algo.name}</span>
          <span className="text-[10px] text-black/40 line-clamp-1">{algo.description}</span>
        </div>
        <ChevronRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity text-emerald-600" />
      </div>
    </button>
  );
};

function MetricCard({ label, value, color }: { label: string, value: string | number, color: 'emerald' | 'stone' }) {
  const colors = {
    emerald: 'bg-emerald-50 text-emerald-900 border-emerald-100',
    stone: 'bg-stone-50 text-stone-900 border-stone-100'
  };
  
  // Format large numbers to scientific notation if they are too long
  const displayValue = typeof value === 'string' && value.length > 15 && !isNaN(parseFloat(value))
    ? parseFloat(value).toExponential(2)
    : value;

  return (
    <div className={`p-6 rounded-3xl border ${colors[color]} shadow-sm flex flex-col justify-center min-w-0`}>
      <span className="text-[10px] font-bold uppercase opacity-50 block mb-1 truncate">{label}</span>
      <span className={`font-bold tracking-tight ${displayValue.toString().length > 10 ? 'text-lg' : 'text-2xl'}`}>
        {displayValue}
      </span>
    </div>
  );
}
