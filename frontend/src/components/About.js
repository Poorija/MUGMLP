import React from 'react';
import { Container, Typography, Paper, Box, Avatar } from '@mui/material';
import { Person, Email } from '@mui/icons-material';
import { useTranslation } from 'react-i18next';

const About = () => {
  const { t } = useTranslation();

  return (
    <Container maxWidth="md" sx={{ mt: 8 }}>
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h4" gutterBottom>{t('about')}</Typography>

        <Box display="flex" flexDirection="column" alignItems="center" mt={4}>
            <Avatar sx={{ width: 80, height: 80, mb: 2, bgcolor: 'primary.main' }}>
                <Person fontSize="large" />
            </Avatar>
            <Typography variant="h5">{t('developer')}</Typography>
            <Typography variant="h6" color="primary">Mohammadmahdi Farhadianfard</Typography>

            <Box display="flex" alignItems="center" gap={1} mt={2}>
                <Email color="action" />
                <Typography variant="body1">mohammadmahdi.farhadianfard@gmail.com</Typography>
            </Box>
        </Box>
      </Paper>
    </Container>
  );
};

export default About;
