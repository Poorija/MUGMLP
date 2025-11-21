import React, { useState } from 'react';
import {
  Dialog, DialogTitle, DialogContent, DialogActions, Button,
  TextField, Grid, Typography, Alert, Box
} from '@mui/material';
import api from '../services/api';

const Predict = ({ model, dataset, open, onClose }) => {
  const [inputData, setInputData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  // Filter out target column from input fields
  const featureColumns = dataset.columns.filter(col => col !== model.target_column);

  const handleChange = (e) => {
    setInputData({ ...inputData, [e.target.name]: e.target.value });
  };

  const handlePredict = async () => {
    setError(null);
    setPrediction(null);
    try {
      // Convert numeric strings to numbers if possible (though backend pipeline might handle it, safer here)
      // For now, sending strings is usually okay with pandas unless schema is strict.
      // We'll just send as is.
      const response = await api.post(`/models/${model.id}/predict`, inputData);
      setPrediction(response.data.predictions);
    } catch (err) {
      setError(err.response?.data?.detail || "Prediction failed");
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Predict with {model.name}</DialogTitle>
      <DialogContent>
        <Grid container spacing={2} sx={{ mt: 1 }}>
            {featureColumns.map(col => (
                <Grid item xs={12} sm={6} key={col}>
                    <TextField
                        fullWidth
                        label={col}
                        name={col}
                        onChange={handleChange}
                        variant="outlined"
                    />
                </Grid>
            ))}
        </Grid>

        {prediction && (
            <Box sx={{ mt: 3, p: 2, bgcolor: '#e8f5e9', borderRadius: 1 }}>
                <Typography variant="h6" color="success.main">Result:</Typography>
                <Typography variant="body1">{JSON.stringify(prediction)}</Typography>
            </Box>
        )}

        {error && (
            <Box sx={{ mt: 3 }}>
                <Alert severity="error">{error}</Alert>
            </Box>
        )}

      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
        <Button onClick={handlePredict} variant="contained" color="primary">Predict</Button>
      </DialogActions>
    </Dialog>
  );
};

export default Predict;
