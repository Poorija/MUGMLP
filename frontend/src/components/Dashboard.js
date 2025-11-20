import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';
import {
  Container, Typography, Grid, Card, CardContent, CardActions,
  Button, TextField, Box, AppBar, Toolbar, IconButton, Paper
} from '@mui/material';
import { Add, Folder, Logout } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const Dashboard = () => {
  const [projects, setProjects] = useState([]);
  const [projectName, setProjectName] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    try {
      const response = await api.get('/projects/');
      setProjects(response.data);
    } catch (err) {
      setError('Failed to fetch projects.');
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

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            ML Platform
          </Typography>
          <Button color="inherit" onClick={handleLogout} startIcon={<Logout />}>Logout</Button>
        </Toolbar>
      </AppBar>

      <Container sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Welcome to Your Dashboard
        </Typography>

        <Paper sx={{ p: 3, mb: 4 }}>
            <Typography variant="h6" gutterBottom>Create New Project</Typography>
            <form onSubmit={handleCreateProject} style={{ display: 'flex', gap: '10px' }}>
                <TextField
                    label="Project Name"
                    variant="outlined"
                    value={projectName}
                    onChange={(e) => setProjectName(e.target.value)}
                    required
                    fullWidth
                />
                <Button variant="contained" type="submit" startIcon={<Add />}>Create</Button>
            </form>
            {error && <Typography color="error" sx={{ mt: 1 }}>{error}</Typography>}
        </Paper>

        <Typography variant="h5" gutterBottom>Your Projects</Typography>
        <Grid container spacing={3}>
            {projects.map((project) => (
            <Grid item xs={12} sm={6} md={4} key={project.id}>
                <Card sx={{ minWidth: 275, '&:hover': { boxShadow: 6 } }}>
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
                    <Button size="small" component={Link} to={`/projects/${project.id}`}>Open Project</Button>
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
