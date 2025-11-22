import React, { useState, useEffect, useContext } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';
import {
  Container, Typography, Grid, Card, CardContent, CardActions,
  Button, TextField, Box, AppBar, Toolbar, IconButton, Paper, Menu, MenuItem, Avatar
} from '@mui/material';
import { Add, Folder, Logout, Brightness4, Brightness7, Translate, Info, AccountCircle } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { ThemeContext } from '../contexts/ThemeContext';
import { Tooltip } from '../GlobalTooltip';
import Logo from './Logo';

const Dashboard = () => {
  const { t, i18n } = useTranslation();
  const { mode, toggleTheme } = useContext(ThemeContext);
  const [projects, setProjects] = useState([]);
  const [projectName, setProjectName] = useState('');
  const [error, setError] = useState('');
  const [anchorEl, setAnchorEl] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    // Check auth
    if (!localStorage.getItem('token')) navigate('/login');
    fetchProjects();
  }, [navigate]);

  const fetchProjects = async () => {
    try {
      const response = await api.get('/projects/');
      setProjects(response.data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleCreateProject = async (e) => {
    e.preventDefault();
    setError('');
    try {
      await api.post('/projects/', { name: projectName });
      setProjectName('');
      fetchProjects();
    } catch (err) {
      setError('Failed to create project.');
    }
  };

  const handleLogout = () => {
      localStorage.removeItem('token');
      navigate('/login');
  };

  const handleLanguageMenu = (event) => setAnchorEl(event.currentTarget);
  const handleLanguageClose = (lang) => {
      if (lang) i18n.changeLanguage(lang);
      setAnchorEl(null);
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" color="default">
        <Toolbar>
          <Logo />
          <Box sx={{ flexGrow: 1 }} />

          {/* Theme Toggle */}
          <IconButton sx={{ ml: 1 }} onClick={toggleTheme} color="inherit">
            {mode === 'dark' ? <Brightness7 /> : <Brightness4 />}
          </IconButton>

          {/* Language Toggle */}
          <IconButton color="inherit" onClick={handleLanguageMenu}>
              <Translate />
          </IconButton>
          <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={() => handleLanguageClose(null)}>
              <MenuItem onClick={() => handleLanguageClose('en')}>English</MenuItem>
              <MenuItem onClick={() => handleLanguageClose('fa')}>فارسی</MenuItem>
          </Menu>

          {/* About */}
          <Tooltip text="About this application">
              <IconButton color="inherit" component={Link} to="/about">
                  <Info />
              </IconButton>
          </Tooltip>

          {/* Profile */}
          <Tooltip text="Manage your profile, password, and security">
              <IconButton color="inherit" component={Link} to="/profile" sx={{ ml: 1 }}>
                   <AccountCircle />
              </IconButton>
          </Tooltip>

          <Button color="inherit" onClick={handleLogout} startIcon={<Logout />} sx={{ ml: 1 }}>{t('logout')}</Button>
        </Toolbar>
      </AppBar>

      <Container sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom fontWeight="bold" className="text-gradient">
          {t('welcome')}
        </Typography>

        <Paper className="glass" elevation={0} sx={{ p: 3, mb: 4 }}>
            <Typography variant="h6" gutterBottom>{t('create_project')}</Typography>
            <form onSubmit={handleCreateProject} style={{ display: 'flex', gap: '10px' }}>
                <Tooltip text="Enter a unique name for your project">
                    <TextField
                        label={t('project_name')}
                        variant="outlined"
                        value={projectName}
                        onChange={(e) => setProjectName(e.target.value)}
                        required
                        fullWidth
                    />
                </Tooltip>
                <Tooltip text="Start a new project workspace">
                    <Button variant="contained" type="submit" startIcon={<Add />}>{t('create')}</Button>
                </Tooltip>
            </form>
            {error && <Typography color="error" sx={{ mt: 1 }}>{error}</Typography>}
        </Paper>

        <Typography variant="h5" gutterBottom fontWeight="bold">{t('projects')}</Typography>
        <Grid container spacing={3}>
            {projects.map((project) => (
            <Grid item xs={12} sm={6} md={4} key={project.id}>
                <Card className="glass" elevation={0} sx={{ minWidth: 275, '&:hover': { transform: 'translateY(-5px)', boxShadow: '0 12px 40px rgba(0,0,0,0.1)' }, transition: 'all 0.3s' }}>
                <CardContent>
                    <Typography variant="h5" component="div">
                    <Folder sx={{ mr: 1, verticalAlign: 'middle' }} />
                    {project.name}
                    </Typography>
                    <Typography sx={{ mb: 1.5 }} color="text.secondary">
                    ID: {project.id}
                    </Typography>
                </CardContent>
                <CardActions>
                    <Button size="small" component={Link} to={`/projects/${project.id}`}>{t('open_project')}</Button>
                </CardActions>
                </Card>
            </Grid>
            ))}
        </Grid>
      </Container>
    </Box>
  );
};

export default Dashboard;
