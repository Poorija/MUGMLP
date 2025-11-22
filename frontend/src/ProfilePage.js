import React, { useState, useEffect } from 'react';
import {
  Container, Typography, Box, Paper, TextField, Button, Grid,
  Avatar, List, ListItem, ListItemText, Divider, Alert, Switch, FormControlLabel
} from '@mui/material';
import axios from 'axios';
import { motion } from 'framer-motion';
import { QRCodeCanvas as QRCode } from 'qrcode.react';

const ProfilePage = () => {
  const [activeTab, setActiveTab] = useState('info'); // info, security, history
  const [user, setUser] = useState({});
  const [passwordData, setPasswordData] = useState({ old_password: '', new_password: '' });
  const [message, setMessage] = useState(null);
  const [history, setHistory] = useState([]);
  const [twoFASecret, setTwoFASecret] = useState('');
  const [twoFAUrl, setTwoFAUrl] = useState('');
  const [twoFACode, setTwoFACode] = useState('');

  useEffect(() => {
    fetchProfile();
    fetchHistory();
  }, []);

  const fetchProfile = async () => {
    try {
      const token = localStorage.getItem('token');
      const res = await axios.get('/users/me', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setUser(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchHistory = async () => {
    try {
      const token = localStorage.getItem('token');
      const res = await axios.get('/users/history', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setHistory(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleUpdateProfile = async () => {
     // TODO: Implement simple email update if needed
     try {
        const token = localStorage.getItem('token');
        await axios.put('/users/me', { email: user.email }, {
            headers: { Authorization: `Bearer ${token}` }
        });
        setMessage({ type: 'success', text: 'Profile updated' });
     } catch (err) {
         setMessage({ type: 'error', text: 'Failed to update profile' });
     }
  };

  const handleChangePassword = async () => {
    try {
      const token = localStorage.getItem('token');
      await axios.post('/auth/change-password', passwordData, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setMessage({ type: 'success', text: 'Password changed successfully' });
      setPasswordData({ old_password: '', new_password: '' });
    } catch (err) {
      setMessage({ type: 'error', text: err.response?.data?.detail || 'Failed to change password' });
    }
  };

  const setup2FA = async () => {
    try {
      const token = localStorage.getItem('token');
      const res = await axios.post('/auth/setup-2fa', {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setTwoFASecret(res.data.secret);
      setTwoFAUrl(res.data.otpauth_url);
    } catch (err) {
      setMessage({ type: 'error', text: 'Failed to setup 2FA' });
    }
  };

  const enable2FA = async () => {
    try {
      const token = localStorage.getItem('token');
      await axios.post('/auth/enable-2fa', { otp_secret: twoFASecret }, {
        params: { code: twoFACode },
        headers: { Authorization: `Bearer ${token}` }
      });
      setMessage({ type: 'success', text: '2FA Enabled!' });
      fetchProfile(); // Refresh to see updated status
      setTwoFASecret('');
    } catch (err) {
      setMessage({ type: 'error', text: 'Invalid Code' });
    }
  };

  const disable2FA = async () => {
     try {
         const token = localStorage.getItem('token');
         await axios.post('/auth/disable-2fa', {}, {
             headers: { Authorization: `Bearer ${token}` }
         });
         setMessage({ type: 'success', text: '2FA Disabled' });
         fetchProfile();
     } catch (err) {
         setMessage({ type: 'error', text: 'Failed to disable 2FA' });
     }
  }

  const glassStyle = {
    background: 'rgba(255, 255, 255, 0.7)',
    backdropFilter: 'blur(20px)',
    borderRadius: '24px',
    boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
    border: '1px solid rgba(255, 255, 255, 0.18)',
    padding: '30px',
    marginBottom: '20px'
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <Paper style={glassStyle} elevation={0}>
          <Box display="flex" alignItems="center" mb={4}>
            <Avatar sx={{ width: 80, height: 80, mr: 3, bgcolor: 'primary.main', fontSize: '2rem' }}>
                {user.email ? user.email[0].toUpperCase() : 'U'}
            </Avatar>
            <Box>
                <Typography variant="h4" fontWeight="bold" gutterBottom>{user.email}</Typography>
                <Typography variant="body2" color="textSecondary">Manage your account settings and security</Typography>
            </Box>
          </Box>

          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
            <Button onClick={() => setActiveTab('info')} sx={{ mr: 2, borderRadius: '20px', px: 3, py: 1, bgcolor: activeTab === 'info' ? 'primary.light' : 'transparent' }}>Profile</Button>
            <Button onClick={() => setActiveTab('security')} sx={{ mr: 2, borderRadius: '20px', px: 3, py: 1, bgcolor: activeTab === 'security' ? 'primary.light' : 'transparent' }}>Security</Button>
            <Button onClick={() => setActiveTab('history')} sx={{ borderRadius: '20px', px: 3, py: 1, bgcolor: activeTab === 'history' ? 'primary.light' : 'transparent' }}>Activity History</Button>
          </Box>

          {message && (
            <Alert severity={message.type} onClose={() => setMessage(null)} sx={{ mb: 3, borderRadius: '16px' }}>
              {message.text}
            </Alert>
          )}

          {activeTab === 'info' && (
             <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <Grid container spacing={3}>
                    <Grid item xs={12}>
                        <TextField
                            fullWidth
                            label="Email"
                            value={user.email || ''}
                            onChange={(e) => setUser({ ...user, email: e.target.value })}
                            variant="outlined"
                            InputProps={{ sx: { borderRadius: '16px' } }}
                        />
                    </Grid>
                    <Grid item xs={12}>
                        <Button variant="contained" onClick={handleUpdateProfile} sx={{ borderRadius: '12px', py: 1.5, px: 4 }}>
                            Save Changes
                        </Button>
                    </Grid>
                </Grid>
             </motion.div>
          )}

          {activeTab === 'security' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <Typography variant="h6" gutterBottom>Change Password</Typography>
              <Grid container spacing={2} sx={{ mb: 4 }}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    type="password"
                    label="Current Password"
                    value={passwordData.old_password}
                    onChange={(e) => setPasswordData({ ...passwordData, old_password: e.target.value })}
                    InputProps={{ sx: { borderRadius: '16px' } }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    type="password"
                    label="New Password"
                    value={passwordData.new_password}
                    onChange={(e) => setPasswordData({ ...passwordData, new_password: e.target.value })}
                    InputProps={{ sx: { borderRadius: '16px' } }}
                  />
                </Grid>
                <Grid item xs={12}>
                  <Button variant="contained" onClick={handleChangePassword} sx={{ borderRadius: '12px' }}>
                    Update Password
                  </Button>
                </Grid>
              </Grid>

              <Divider sx={{ my: 4 }} />

              <Typography variant="h6" gutterBottom>Two-Factor Authentication (2FA)</Typography>

              {user.otp_secret ? (
                  <Box>
                      <Alert severity="success" sx={{ borderRadius: '16px', mb: 2 }}>2FA is currently enabled.</Alert>
                      <Button variant="outlined" color="error" onClick={disable2FA} sx={{ borderRadius: '12px' }}>Disable 2FA</Button>
                  </Box>
              ) : (
                  <Box>
                    {!twoFASecret ? (
                         <Button variant="contained" onClick={setup2FA} sx={{ borderRadius: '12px' }}>Setup 2FA</Button>
                    ) : (
                        <Box sx={{ textAlign: 'center', p: 3, bgcolor: 'background.paper', borderRadius: '20px' }}>
                            <Typography gutterBottom>Scan this QR Code with your Authenticator App</Typography>
                            <Box sx={{ my: 2 }}>
                                <QRCode value={twoFAUrl} size={200} />
                            </Box>
                            <TextField
                                label="Verification Code"
                                value={twoFACode}
                                onChange={(e) => setTwoFACode(e.target.value)}
                                sx={{ mb: 2, width: '200px' }}
                                InputProps={{ sx: { borderRadius: '16px' } }}
                            />
                            <br/>
                            <Button variant="contained" onClick={enable2FA} sx={{ borderRadius: '12px' }}>Verify & Enable</Button>
                        </Box>
                    )}
                  </Box>
              )}
            </motion.div>
          )}

          {activeTab === 'history' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <List>
                {history.map((log) => (
                  <ListItem key={log.id} divider>
                    <ListItemText
                      primary={log.action}
                      secondary={
                        <>
                            <Typography variant="body2" component="span">{new Date(log.timestamp).toLocaleString()}</Typography>
                            <br/>
                            {log.details}
                        </>
                      }
                    />
                  </ListItem>
                ))}
                {history.length === 0 && <Typography color="textSecondary">No activity recorded yet.</Typography>}
              </List>
            </motion.div>
          )}

        </Paper>
      </motion.div>
    </Container>
  );
};

export default ProfilePage;
