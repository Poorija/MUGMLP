import React, { useState, useEffect } from 'react';
import api from '../services/api';

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
      fetchCaptcha(); // Get a new captcha after a failed attempt
    }
  };

  return (
    <div>
      <h2>Register</h2>
      <form onSubmit={handleSubmit}>
        <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} required />

        {captchaImageUrl && (
          <div>
            <img src={captchaImageUrl} alt="Captcha" />
            <button type="button" onClick={fetchCaptcha}>Refresh Captcha</button>
            <input
              type="text"
              placeholder="Enter Captcha"
              value={captchaText}
              onChange={(e) => setCaptchaText(e.target.value)}
              required
            />
          </div>
        )}

        <button type="submit">Register</button>
      </form>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {success && <p style={{ color: 'green' }}>{success}</p>}
    </div>
  );
};

export default Register;
