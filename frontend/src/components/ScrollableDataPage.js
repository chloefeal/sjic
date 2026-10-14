import React from 'react';
import { Box, Card, CardContent } from '@mui/material';

/** 表格区域单独滚动，配合 Table stickyHeader 让列标题留在滚动区顶部。 */
export const scrollTableContainerSx = {
  flex: 1,
  minHeight: 0,
  overflow: 'auto',
};

function ScrollableDataPage({ header, footer, children, variant = 'plain' }) {
  const body = (
    <>
      {header ? <Box sx={{ flexShrink: 0 }}>{header}</Box> : null}
      <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        {children}
      </Box>
      {footer ? <Box sx={{ flexShrink: 0 }}>{footer}</Box> : null}
    </>
  );

  if (variant === 'card') {
    return (
      <Card sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <CardContent
          sx={{
            flex: 1,
            minHeight: 0,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            '&:last-child': { pb: 2 },
          }}
        >
          {body}
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {body}
    </Box>
  );
}

export default ScrollableDataPage;
