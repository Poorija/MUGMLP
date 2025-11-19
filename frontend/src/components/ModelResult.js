import React from 'react';

const ModelResult = ({ model }) => {
  if (!model) return null;

  const metrics = model.evaluation_metrics ? JSON.parse(model.evaluation_metrics) : {};
  const hyperparameters = model.hyperparameters ? JSON.parse(model.hyperparameters) : {};

  return (
    <div style={{ border: '1px solid #007bff', padding: '10px', marginTop: '10px' }}>
      <h3>Results for: {model.name}</h3>
      <p><strong>Status:</strong> {model.status}</p>
      <p><strong>Model Type:</strong> {model.model_type}</p>
      <p><strong>Task Type:</strong> {model.task_type}</p>

      <h4>Evaluation Metrics</h4>
      {Object.keys(metrics).length > 0 ? (
        <ul>
          {Object.entries(metrics).map(([key, value]) => (
            <li key={key}><strong>{key}:</strong> {JSON.stringify(value)}</li>
          ))}
        </ul>
      ) : (
        <p>No metrics available (model may be pending, running, or failed).</p>
      )}

      <h4>Visualizations</h4>
      {model.status === 'completed' && (
        <div>
          {model.task_type === 'classification' && (
            <div>
              <h5>Confusion Matrix</h5>
              <img src={`http://localhost:8000/models/${model.id}/visualizations/confusion_matrix`} alt="Confusion Matrix" />
            </div>
          )}
          {model.task_type === 'clustering' && (
            <div>
              <h5>Cluster Scatter Plot</h5>
              <img src={`http://localhost:8000/models/${model.id}/visualizations/cluster_scatter`} alt="Cluster Plot" />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelResult;
