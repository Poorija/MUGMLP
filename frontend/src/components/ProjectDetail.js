import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import api from '../services/api';
import ModelResult from './ModelResult'; // Import the new component

// TrainingForm component remains the same as the previous step...
const TASK_DEFINITIONS = {
  classification: {
    models: ["KNeighborsClassifier", "DecisionTreeClassifier", "RandomForestClassifier", "XGBClassifier", "SimpleNN"],
    requiresTarget: true,
  },
  clustering: {
    models: ["KMeans", "DBSCAN"],
    requiresTarget: false,
  },
  dimensionality_reduction: {
    models: ["PCA"],
    requiresTarget: false,
  }
};
const TrainingForm = ({ dataset, onTrainingStart }) => {
  const [taskType, setTaskType] = useState('classification');
  const [modelType, setModelType] = useState(TASK_DEFINITIONS.classification.models[0]);
  const [modelName, setModelName] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  useEffect(() => {
    setModelType(TASK_DEFINITIONS[taskType].models[0]);
  }, [taskType]);
  const handleSubmit = async (e) => {
    e.preventDefault();
    const payload = {
      name: modelName,
      task_type: taskType,
      model_type: modelType,
      target_column: TASK_DEFINITIONS[taskType].requiresTarget ? targetColumn : null,
      hyperparameters: { /* TODO: Add dynamic hyperparameters */ }
    };
    try {
      await api.post(`/datasets/${dataset.id}/train`, payload);
      onTrainingStart();
    } catch (error) {
      alert('Failed to start training.');
    }
  };
  return (
    <form onSubmit={handleSubmit} style={{ border: '1px solid #ccc', padding: '10px', marginTop: '10px' }}>
      <h4>Start New Training</h4>
      <input type="text" placeholder="Model Name" value={modelName} onChange={e => setModelName(e.target.value)} required />
      <select value={taskType} onChange={e => setTaskType(e.target.value)}>
        {Object.keys(TASK_DEFINITIONS).map(task => <option key={task} value={task}>{task}</option>)}
      </select>
      <select value={modelType} onChange={e => setModelType(e.target.value)}>
        {TASK_DEFINITIONS[taskType].models.map(model => <option key={model} value={model}>{model}</option>)}
      </select>
      {TASK_DEFINITIONS[taskType].requiresTarget && (
        <select value={targetColumn} onChange={e => setTargetColumn(e.target.value)} required>
          <option value="">Select Target Column</option>
          {dataset.columns.map(col => <option key={col} value={col}>{col}</option>)}
        </select>
      )}
      <button type="submit">Train Model</button>
    </form>
  );
};


const ProjectDetail = () => {
  const { projectId } = useParams();
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [mlModels, setMlModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);

  // Fetch datasets on component mount
  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await api.get(`/projects/${projectId}/datasets`);
        setDatasets(response.data);
      } catch (err) { console.error(err); }
    };
    fetchDatasets();
  }, [projectId]);

  // Fetch models when a dataset is selected
  const fetchModels = async (datasetId) => {
    try {
      const response = await api.get(`/datasets/${datasetId}/models`);
      setMlModels(response.data);
    } catch (err) {
      console.error("Failed to fetch models", err);
    }
  };

  const handleDatasetSelect = (dataset) => {
    const metadata = JSON.parse(dataset.metadata);
    setSelectedDataset({ ...dataset, columns: metadata.columns });
    setSelectedModel(null); // Reset selected model
    fetchModels(dataset.id);
  };

  const handleTrainingStart = () => {
    alert('Training started! The model list will update in a moment.');
    // Refresh model list after a short delay
    setTimeout(() => {
      if (selectedDataset) {
        fetchModels(selectedDataset.id);
      }
    }, 5000); // 5 seconds delay
  };

  return (
    <div>
      <h2>Project Details (ID: {projectId})</h2>
      <h3>Datasets</h3>
      <ul>
        {datasets.map((ds) => (
          <li key={ds.id} onClick={() => handleDatasetSelect(ds)} style={{ cursor: 'pointer', fontWeight: selectedDataset?.id === ds.id ? 'bold' : 'normal' }}>
            {ds.filename}
          </li>
        ))}
      </ul>

      {selectedDataset && (
        <div>
          <TrainingForm
            dataset={selectedDataset}
            onTrainingStart={handleTrainingStart}
          />

          <div style={{ marginTop: '20px' }}>
            <h3>Trained Models for {selectedDataset.filename}</h3>
            <ul>
              {mlModels.map(model => (
                <li key={model.id} onClick={() => setSelectedModel(model)} style={{ cursor: 'pointer', fontWeight: selectedModel?.id === model.id ? 'bold' : 'normal' }}>
                  {model.name} ({model.model_type}) - Status: {model.status}
                </li>
              ))}
            </ul>
          </div>

          <ModelResult model={selectedModel} />
        </div>
      )}
    </div>
  );
};

export default ProjectDetail;
