import React, { createContext, useContext, useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const TooltipContext = createContext();

export const useTooltip = () => useContext(TooltipContext);

export const TooltipProvider = ({ children }) => {
  const [tooltip, setTooltip] = useState({ visible: false, text: '', x: 0, y: 0 });

  const showTooltip = (text, e) => {
    setTooltip({
      visible: true,
      text,
      x: e.clientX,
      y: e.clientY + 15, // Offset
    });
  };

  const hideTooltip = () => {
    setTooltip((prev) => ({ ...prev, visible: false }));
  };

  const updatePosition = (e) => {
    if (tooltip.visible) {
        setTooltip(prev => ({ ...prev, x: e.clientX, y: e.clientY + 15 }))
    }
  }

  // Handle global mouse move to update tooltip position if needed,
  // though typically we attach events to elements.
  // For "following" cursor tooltips:
  useEffect(() => {
    if(tooltip.visible){
        window.addEventListener('mousemove', updatePosition)
    }
    return () => window.removeEventListener('mousemove', updatePosition)
  }, [tooltip.visible])


  return (
    <TooltipContext.Provider value={{ showTooltip, hideTooltip }}>
      {children}
      <AnimatePresence>
        {tooltip.visible && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            style={{
              position: 'fixed',
              left: tooltip.x,
              top: tooltip.y,
              backgroundColor: 'rgba(25, 25, 35, 0.9)',
              backdropFilter: 'blur(10px)',
              color: '#fff',
              padding: '8px 12px',
              borderRadius: '8px',
              fontSize: '12px',
              pointerEvents: 'none',
              zIndex: 9999,
              boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
              border: '1px solid rgba(255,255,255,0.1)',
            }}
          >
            {tooltip.text}
          </motion.div>
        )}
      </AnimatePresence>
    </TooltipContext.Provider>
  );
};

// Component to wrap elements that need tooltips
export const Tooltip = ({ text, children }) => {
  const { showTooltip, hideTooltip } = useTooltip();

  return (
    <div
      onMouseEnter={(e) => showTooltip(text, e)}
      onMouseLeave={hideTooltip}
      style={{ display: 'inline-block' }}
    >
      {children}
    </div>
  );
};
