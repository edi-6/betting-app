import React, { useMemo } from 'react';
import { View } from 'react-native';
import Svg, { Path } from 'react-native-svg';

import { useTheme } from '../../theme';

export interface SparklineProps {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
}

/** Tiny inline trend line — no axes, no interaction. */
export function Sparkline({ values, width = 72, height = 24, color }: SparklineProps) {
  const theme = useTheme();
  const stroke = color ?? theme.colors.primary;

  const path = useMemo(() => {
    if (values.length < 2) {
      return '';
    }
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    return values
      .map((value, index) => {
        const x = (index / (values.length - 1)) * width;
        const y = height - ((value - min) / span) * height;
        return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(' ');
  }, [values, width, height]);

  if (path.length === 0) {
    return <View style={{ width, height }} />;
  }

  return (
    <Svg width={width} height={height}>
      <Path d={path} stroke={stroke} strokeWidth={2} fill="none" strokeLinejoin="round" strokeLinecap="round" />
    </Svg>
  );
}
