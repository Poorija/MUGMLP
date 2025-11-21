import React, { useState, useEffect } from 'react';
import api from '../services/api';
import { Container, Typography, TextField, Button, Paper, Box, Alert } from '@mui/material';
import { Refresh } from '@mui/icons-material';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import zxcvbn from 'zxcvbn';

const Register = () => {
  const { t } = useTranslation();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordStrength, setPasswordStrength] = useState(0);
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

  const handlePasswordChange = (e) => {
      const val = e.target.value;
      setPassword(val);
      const result = zxcvbn(val);
      setPasswordStrength(result.score);
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
      setError(err.response?.data?.detail || 'Registration failed.');
      fetchCaptcha();
    }
  };

  const strengthLabel = [t('weak'), t('weak'), t('fair'), t('good'), t('strong')];
  const strengthColor = ['error', 'error', 'warning', 'success', 'success'];

  return (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
      <Paper sx={{ p: 4 }}>
        <Typography variant="h4" align="center" gutterBottom>{t('register')}</Typography>
        <form onSubmit={handleSubmit}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <TextField
                label={t('email')}
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                fullWidth
              />
              <TextField
                label={t('password')}
                type="password"
                value={password}
                onChange={handlePasswordChange}
                required
                fullWidth
              />

              {/* Password Strength */}
              <Box>
                    <Typography variant="caption">{t('password_strength')}: </Typography>
                    <Typography variant="caption" color={`${strengthColor[passwordStrength]}.main`} fontWeight="bold">
                        {strengthLabel[passwordStrength]}
                    </Typography>
                    <div style={{ height: 5, width: '100%', backgroundColor: '#e0e0e0', borderRadius: 2, marginTop: 4 }}>
                        <div style={{
                            height: '100%',
                            width: `${(passwordStrength + 1) * 20}%`,
                            backgroundColor: passwordStrength < 2 ? 'red' : passwordStrength === 2 ? 'orange' : 'green',
                            borderRadius: 2,
                            transition: 'width 0.3s'
                        }} />
                    </div>
              </Box>

              {captchaImageUrl && (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1, border: '1px solid #eee', p: 2, borderRadius: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <img src={captchaImageUrl} alt="Captcha" style={{ maxHeight: 50 }} />
                        <Button onClick={fetchCaptcha} startIcon={<Refresh />}>{t('captcha_refresh')}</Button>
                    </Box>
                    <TextField
                        label={t('captcha')}
                        value={captchaText}
                        onChange={(e) => setCaptchaText(e.target.value)}
                        required
                        fullWidth
                    />
                </Box>
              )}

              <Button type="submit" variant="contained" size="large" fullWidth disabled={passwordStrength < 2}>{t('register')}</Button>
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
