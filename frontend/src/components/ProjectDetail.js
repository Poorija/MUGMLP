import React, { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import api from '../services/api';
import ModelResult from './ModelResult';
import {
  Container, Typography, Tabs, Tab, Box, Paper, List, ListItem, ListItemText,
  Grid, Card, CardContent, Button, Select, MenuItem, FormControl, InputLabel,
  TextField, CircularProgress, Alert
} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line
} from 'recharts';
import { CloudUpload, PlayArrow } from '@mui/icons-material';

// Task definitions including Regression now
const TASK_DEFINITIONS = {
  classification: {
    models: ["KNeighborsClassifier", "DecisionTreeClassifier", "RandomForestClassifier", "XGBClassifier", "SimpleNN"],
    requiresTarget: true,
  },
  regression: {
      models: ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "XGBRegressor", "SVR", "SimpleNN"],
      requiresTarget: true
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

function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const TrainingForm = ({ dataset, onTrainingStart }) => {
  const [taskType, setTaskType] = useState('classification');
  const [modelType, setModelType] = useState(TASK_DEFINITIONS.classification.models[0]);
  const [modelName, setModelName] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [hyperparams, setHyperparams] = useState({});

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
      hyperparameters: hyperparams
    };
    try {
      await api.post(`/datasets/${dataset.id}/train`, payload);
      onTrainingStart();
    } catch (error) {
      alert('Failed to start training.');
    }
  };

  return (
    <Paper sx={{ p: 2, mt: 2 }}>
      <Typography variant="h6" gutterBottom>Start New Training</Typography>
      <form onSubmit={handleSubmit}>
        <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
                <TextField
                    fullWidth label="Model Name"
                    value={modelName}
                    onChange={e => setModelName(e.target.value)}
                    required
                />
            </Grid>
            <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                    <InputLabel>Task Type</InputLabel>
                    <Select value={taskType} label="Task Type" onChange={e => setTaskType(e.target.value)}>
                        {Object.keys(TASK_DEFINITIONS).map(task => <MenuItem key={task} value={task}>{task}</MenuItem>)}
                    </Select>
                </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                    <InputLabel>Model Type</InputLabel>
                    <Select value={modelType} label="Model Type" onChange={e => setModelType(e.target.value)}>
                        {TASK_DEFINITIONS[taskType].models.map(model => <MenuItem key={model} value={model}>{model}</MenuItem>)}
                    </Select>
                </FormControl>
            </Grid>
            {TASK_DEFINITIONS[taskType].requiresTarget && (
                <Grid item xs={12} sm={6}>
                    <FormControl fullWidth required>
                        <InputLabel>Target Column</InputLabel>
                        <Select value={targetColumn} label="Target Column" onChange={e => setTargetColumn(e.target.value)}>
                             {dataset.columns.map(col => <MenuItem key={col} value={col}>{col}</MenuItem>)}
                        </Select>
                    </FormControl>
                </Grid>
            )}
            {/* SimpleNN specific params */}
            {modelType === 'SimpleNN' && (
                <>
                    <Grid item xs={12} sm={6}>
                         <TextField fullWidth label="Hidden Layers (e.g., 64,32)" onChange={e => setHyperparams({...hyperparams, hidden_layers: e.target.value})} />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                         <TextField fullWidth label="Epochs" type="number" onChange={e => setHyperparams({...hyperparams, epochs: e.target.value})} />
                    </Grid>
                </>
            )}
            <Grid item xs={12}>
                <Button type="submit" variant="contained" startIcon={<PlayArrow />}>Train Model</Button>
            </Grid>
        </Grid>
      </form>
    </Paper>
  );
};

const ProjectDetail = () => {
  const { projectId } = useParams();
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetData, setDatasetData] = useState([]);
  const [datasetSummary, setDatasetSummary] = useState(null);
  const [mlModels, setMlModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [trainingProgress, setTrainingProgress] = useState({});

  const ws = useRef(null);

  // Fetch datasets
  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await api.get(`/projects/${projectId}/datasets`);
        setDatasets(response.data);
      } catch (err) { console.error(err); }
    };
    fetchDatasets();
  }, [projectId]);

  // WebSocket Setup
  useEffect(() => {
      ws.current = new WebSocket(`ws://${window.location.hostname}:8000/ws`);

      ws.current.onopen = () => console.log("WebSocket Connected");
      ws.current.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.status === "running" && data.epoch) {
              setTrainingProgress(prev => ({
                  ...prev,
                  [data.model_id]: { epoch: data.epoch, total: data.total_epochs, loss: data.loss }
              }));
          } else if (data.status === "completed" || data.status === "failed") {
             setTrainingProgress(prev => {
                 const newState = { ...prev };
                 delete newState[data.model_id];
                 return newState;
             });
             // Refresh models list if current dataset is selected
             if (selectedDataset) fetchModels(selectedDataset.id);
          }
      };

      return () => {
          if (ws.current) ws.current.close();
      }
  }, [selectedDataset]); // Re-bind if needed, though usually once is fine. simpler to keep consistent.

  const fetchModels = async (datasetId) => {
    try {
      const response = await api.get(`/datasets/${datasetId}/models`);
      setMlModels(response.data);
    } catch (err) { console.error("Failed to fetch models", err); }
  };

  const fetchDatasetDetails = async (dataset) => {
      try {
          const dataRes = await api.get(`/datasets/${dataset.id}`);
          setDatasetData(dataRes.data);
          const summaryRes = await api.get(`/datasets/${dataset.id}/summary`);
          setDatasetSummary(summaryRes.data);
      } catch(e) { console.error(e); }
  }

  const handleDatasetSelect = (dataset) => {
    const metadata = JSON.parse(dataset.metadata);
    setSelectedDataset({ ...dataset, columns: metadata.columns });
    setSelectedModel(null);
    setDatasetSummary(null); // clear prev
    fetchModels(dataset.id);
    fetchDatasetDetails(dataset);
    setTabValue(0); // Go to dashboard tab
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleFileUpload = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const formData = new FormData();
      formData.append('file', file);
      try {
          await api.post(`/projects/${projectId}/datasets/`, formData, {
              headers: { 'Content-Type': 'multipart/form-data' }
          });
          // Refresh datasets
          const response = await api.get(`/projects/${projectId}/datasets`);
          setDatasets(response.data);
      } catch(e) { alert('Upload failed'); }
  };

  // Comparison Data Preparation
  const getComparisonData = () => {
      if (mlModels.length === 0) return [];
      // Filter for models that have metrics
      const completedModels = mlModels.filter(m => m.status === 'completed' && m.evaluation_metrics);
      return completedModels.map(m => {
          const metrics = JSON.parse(m.evaluation_metrics);
          return {
              name: m.name,
              ...metrics
          };
      });
  };
  const comparisonData = getComparisonData();
  const metricKeys = comparisonData.length > 0 ? Object.keys(comparisonData[0]).filter(k => k !== 'name') : [];

  return (
    <Container maxWidth="xl" sx={{ mt: 4 }}>
        <Button component={Link} to="/dashboard" sx={{ mb: 2 }}>&larr; Back to Dashboard</Button>
        <Grid container spacing={3}>
            {/* Left Sidebar: Datasets */}
            <Grid item xs={12} md={3}>
                <Paper sx={{ p: 2, height: '100%' }}>
                    <Typography variant="h6">Datasets</Typography>
                    <List>
                        {datasets.map((ds) => (
                            <ListItem button key={ds.id} selected={selectedDataset?.id === ds.id} onClick={() => handleDatasetSelect(ds)}>
                                <ListItemText primary={ds.filename} />
                            </ListItem>
                        ))}
                    </List>
                    <Button variant="contained" component="label" fullWidth startIcon={<CloudUpload />}>
                        Upload Dataset
                        <input type="file" hidden onChange={handleFileUpload} accept=".csv,.xlsx,.xls" />
                    </Button>
                </Paper>
            </Grid>

            {/* Main Content */}
            <Grid item xs={12} md={9}>
                {selectedDataset ? (
                    <Paper sx={{ width: '100%' }}>
                        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                            <Tabs value={tabValue} onChange={handleTabChange}>
                                <Tab label="Overview" />
                                <Tab label="Data Analysis" />
                                <Tab label="Models" />
                                <Tab label="Comparison" />
                            </Tabs>
                        </Box>

                        {/* Overview Tab */}
                        <TabPanel value={tabValue} index={0}>
                            <Typography variant="h5">{selectedDataset.filename}</Typography>
                            <Typography variant="body1">Columns: {selectedDataset.columns.join(', ')}</Typography>
                            <TrainingForm dataset={selectedDataset} onTrainingStart={() => setTabValue(2)} />
                        </TabPanel>

                        {/* Data Analysis Tab */}
                        <TabPanel value={tabValue} index={1}>
                             {datasetSummary && (
                                 <Grid container spacing={2}>
                                     <Grid item xs={12}>
                                         <Typography variant="h6">Distributions</Typography>
                                     </Grid>
                                     {Object.entries(datasetSummary.distributions || {}).map(([col, data]) => (
                                         <Grid item xs={12} md={6} key={col}>
                                             <Card>
                                                 <CardContent>
                                                     <Typography>{col}</Typography>
                                                     <ResponsiveContainer width="100%" height={200}>
                                                         <BarChart data={data}>
                                                             <CartesianGrid strokeDasharray="3 3" />
                                                             <XAxis dataKey="bin" />
                                                             <YAxis />
                                                             <Tooltip />
                                                             <Bar dataKey="count" fill="#8884d8" />
                                                         </BarChart>
                                                     </ResponsiveContainer>
                                                 </CardContent>
                                             </Card>
                                         </Grid>
                                     ))}
                                 </Grid>
                             )}
                        </TabPanel>

                        {/* Models Tab */}
                        <TabPanel value={tabValue} index={2}>
                            <Grid container spacing={2}>
                                <Grid item xs={4}>
                                    <List>
                                        {mlModels.map(model => (
                                            <ListItem button key={model.id} selected={selectedModel?.id === model.id} onClick={() => setSelectedModel(model)}>
                                                <ListItemText
                                                    primary={model.name}
                                                    secondary={`${model.model_type} - ${model.status}`}
                                                />
                                                {trainingProgress[model.id] && (
                                                    <CircularProgress variant="determinate" value={(trainingProgress[model.id].epoch / trainingProgress[model.id].total) * 100} size={20} />
                                                )}
                                            </ListItem>
                                        ))}
                                    </List>
                                </Grid>
                                <Grid item xs={8}>
                                    {selectedModel ? (
                                        <ModelResult model={selectedModel} dataset={selectedDataset} />
                                    ) : (
                                        <Typography>Select a model to view results.</Typography>
                                    )}
                                </Grid>
                            </Grid>
                        </TabPanel>

                        {/* Comparison Tab */}
                        <TabPanel value={tabValue} index={3}>
                            <Typography variant="h6">Model Comparison</Typography>
                            {comparisonData.length > 0 ? (
                                <div style={{ height: 400 }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={comparisonData}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="name" />
                                            <YAxis />
                                            <Tooltip />
                                            <Legend />
                                            {metricKeys.map((key, idx) => (
                                                <Bar key={key} dataKey={key} fill={['#8884d8', '#82ca9d', '#ffc658'][idx % 3]} />
                                            ))}
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            ) : <Typography>No completed models to compare.</Typography>}
                        </TabPanel>

                    </Paper>
                ) : (
                    <Box sx={{ p: 3, textAlign: 'center' }}>
                        <Typography variant="h5" color="text.secondary">Select a dataset to begin</Typography>
                    </Box>
                )}
            </Grid>
        </Grid>
    </Container>
  );
};

export default ProjectDetail;
