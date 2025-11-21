import React from 'react';
import { Typography, Box } from '@mui/material';
import { Grain } from '@mui/icons-material'; // Icon that looks a bit like ML nodes

const Logo = ({ size = 'medium' }) => {
  const fontSize = size === 'large' ? 40 : 24;
  const iconSize = size === 'large' ? 50 : 30;

  return (
    <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
        <Grain sx={{ fontSize: iconSize, color: 'primary.main' }} />
        <Typography variant="h6" component="div" sx={{ fontSize, fontWeight: 'bold', background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            ML Platform
        </Typography>
    </Box>
  );
};

export default Logo;
