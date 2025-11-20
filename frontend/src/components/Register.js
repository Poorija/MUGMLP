import React, { useState, useEffect } from 'react';
import api from '../services/api';
import { Container, Typography, TextField, Button, Paper, Box, Alert } from '@mui/material';
import { Refresh } from '@mui/icons-material';
import { Link } from 'react-router-dom';

const Register = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [captchaText, setCaptchaText] = useState('');
  const [captchaImageUrl, setCaptchaImageUrl] = useState('');
  const [captchaSessionId, setCaptchaSessionId] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  useEffect(() => {
    fetchCaptcha();
  }, []);

  const fetchCaptcha = async () => {
    try {
      const response = await api.get('/captcha', { responseType: 'blob' });
      const imageUrl = URL.createObjectURL(response.data);
      const sessionId = response.headers['x-captcha-session-id'];
      setCaptchaImageUrl(imageUrl);
      setCaptchaSessionId(sessionId);
    } catch (err) {
      setError('Failed to load captcha. Please refresh the page.');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    try {
      await api.post('/users/', {
        email,
        password,
        captcha_session_id: captchaSessionId,
        captcha_text: captchaText,
      });
      setSuccess('Registration successful! You can now login.');
    } catch (err) {
      setError('Registration failed. Invalid captcha or email already in use.');
      fetchCaptcha();
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
      <Paper sx={{ p: 4 }}>
        <Typography variant="h4" align="center" gutterBottom>Register</Typography>
        <form onSubmit={handleSubmit}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <TextField
                label="Email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                fullWidth
              />
              <TextField
                label="Password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                fullWidth
              />

              {captchaImageUrl && (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1, border: '1px solid #eee', p: 2, borderRadius: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <img src={captchaImageUrl} alt="Captcha" style={{ maxHeight: 50 }} />
                        <Button onClick={fetchCaptcha} startIcon={<Refresh />}>Refresh</Button>
                    </Box>
                    <TextField
                        label="Enter Captcha"
                        value={captchaText}
                        onChange={(e) => setCaptchaText(e.target.value)}
                        required
                        fullWidth
                    />
                </Box>
              )}

              <Button type="submit" variant="contained" size="large" fullWidth>Register</Button>
          </Box>
        </form>

        {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
        {success && (
            <Box sx={{ mt: 2 }}>
                <Alert severity="success">{success}</Alert>
                <Button component={Link} to="/login" sx={{ mt: 1 }}>Go to Login</Button>
            </Box>
        )}
      </Paper>
    </Container>
  );
};

export default Register;
