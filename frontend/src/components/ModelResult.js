import React, { useState } from 'react';
import { Card, CardContent, Typography, Chip, Box, Grid, Button } from '@mui/material';
import { CheckCircle, Error, HourglassEmpty, Loop, Science } from '@mui/icons-material';
import Predict from './Predict';

const ModelResult = ({ model, dataset }) => {
  const [predictOpen, setPredictOpen] = useState(false);
  if (!model) return null;

  const metrics = model.evaluation_metrics ? JSON.parse(model.evaluation_metrics) : {};
  const hyperparameters = model.hyperparameters ? JSON.parse(model.hyperparameters) : {};

  const getStatusIcon = (status) => {
      switch(status) {
          case 'completed': return <CheckCircle color="success" />;
          case 'failed': return <Error color="error" />;
          case 'running': return <Loop className="spin" color="primary" />; // Need to add spin css or use CircularProgress
          default: return <HourglassEmpty color="action" />;
      }
  };

  return (
    <Box>
        <Card variant="outlined" sx={{ mb: 2 }}>
            <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                    <Typography variant="h5">{model.name}</Typography>
                    <Box>
                        {model.status === 'completed' && (
                            <Button
                                variant="outlined"
                                startIcon={<Science />}
                                size="small"
                                onClick={() => setPredictOpen(true)}
                                sx={{ mr: 1 }}
                            >
                                Predict
                            </Button>
                        )}
                        <Chip icon={getStatusIcon(model.status)} label={model.status.toUpperCase()} />
                    </Box>
                </Box>
                <Typography color="text.secondary" gutterBottom>{model.model_type} ({model.task_type})</Typography>

                <Grid container spacing={2} sx={{ mt: 1 }}>
                    <Grid item xs={12} md={6}>
                        <Typography variant="h6" fontSize={16}>Hyperparameters</Typography>
                        {Object.entries(hyperparameters).map(([k, v]) => (
                            <Typography key={k} variant="body2"><strong>{k}:</strong> {v}</Typography>
                        ))}
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <Typography variant="h6" fontSize={16}>Metrics</Typography>
                        {Object.keys(metrics).length > 0 ? Object.entries(metrics).map(([k, v]) => (
                            <Typography key={k} variant="body2"><strong>{k}:</strong> {typeof v === 'number' ? v.toFixed(4) : v}</Typography>
                        )) : <Typography variant="body2">No metrics yet.</Typography>}
                    </Grid>
                </Grid>
            </CardContent>
        </Card>

        {model.status === 'completed' && (
            <Grid container spacing={2}>
                {model.task_type === 'classification' && (
                    <Grid item xs={12} md={6}>
                        <Typography variant="h6">Confusion Matrix</Typography>
                        <img
                            src={`http://localhost:8000/models/${model.id}/visualizations/confusion_matrix`}
                            alt="Confusion Matrix"
                            style={{ maxWidth: '100%', height: 'auto', border: '1px solid #ddd' }}
                        />
                    </Grid>
                )}
                {model.task_type === 'clustering' && (
                    <Grid item xs={12} md={6}>
                        <Typography variant="h6">Cluster Scatter Plot</Typography>
                        <img
                            src={`http://localhost:8000/models/${model.id}/visualizations/cluster_scatter`}
                            alt="Cluster Plot"
                            style={{ maxWidth: '100%', height: 'auto', border: '1px solid #ddd' }}
                        />
                    </Grid>
                )}
                {model.task_type === 'regression' && (
                    <Grid item xs={12} md={6}>
                        <Typography variant="h6">Actual vs Predicted</Typography>
                        <img
                            src={`http://localhost:8000/models/${model.id}/visualizations/actual_vs_predicted`}
                            alt="Actual vs Predicted"
                            style={{ maxWidth: '100%', height: 'auto', border: '1px solid #ddd' }}
                        />
                    </Grid>
                )}
                {['RandomForestClassifier', 'DecisionTreeClassifier', 'RandomForestRegressor', 'DecisionTreeRegressor', 'XGBClassifier', 'XGBRegressor'].includes(model.model_type) && (
                    <Grid item xs={12} md={6}>
                        <Typography variant="h6">Feature Importance</Typography>
                        <img
                            src={`http://localhost:8000/models/${model.id}/visualizations/feature_importance`}
                            alt="Feature Importance"
                            style={{ maxWidth: '100%', height: 'auto', border: '1px solid #ddd' }}
                        />
                    </Grid>
                )}
            </Grid>
        )}

        <Predict
            open={predictOpen}
            onClose={() => setPredictOpen(false)}
            model={model}
            dataset={dataset}
        />
    </Box>
  );
};

export default ModelResult;
