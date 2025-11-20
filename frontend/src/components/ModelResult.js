import React from 'react';
import { Card, CardContent, Typography, Chip, Box, Grid } from '@mui/material';
import { CheckCircle, Error, HourglassEmpty, Loop } from '@mui/icons-material';

const ModelResult = ({ model }) => {
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
                    <Chip icon={getStatusIcon(model.status)} label={model.status.toUpperCase()} />
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
            </Grid>
        )}
    </Box>
  );
};

export default ModelResult;
