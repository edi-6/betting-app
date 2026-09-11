import React, { useCallback, useMemo, useState } from 'react';
import { StyleSheet, View, type GestureResponderEvent } from 'react-native';
import Svg, { Circle, Defs, Line, LinearGradient, Path, Stop } from 'react-native-svg';

import { useTheme } from '../../theme';
import { Text } from '../Text';

export interface LinePoint {
  x: number;
  y: number;
  label?: string;
}

export interface LineChartProps {
  data: LinePoint[];
  height?: number;
  color?: string;
  /** Draw a dashed baseline at y = 0 (useful for profit curves). */
  showZeroLine?: boolean;
  /** Paint the area below the line. */
  filled?: boolean;
  formatValue?: (value: number) => string;
  formatLabel?: (point: LinePoint) => string;
  emptyMessage?: string;
  testID?: string;
}

/**
 * Compact time-series chart. Touch anywhere to scrub: the nearest point is highlighted
 * and its value is shown above the plot.
 */
export function LineChart({
  data,
  height = 180,
  color,
  showZeroLine = true,
  filled = true,
  formatValue = (value) => value.toFixed(2),
  formatLabel,
  emptyMessage = 'No data yet.',
  testID,
}: LineChartProps) {
  const theme = useTheme();
  const [width, setWidth] = useState(0);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  const stroke = color ?? theme.colors.primary;
  const padding = { top: 12, right: 8, bottom: 12, left: 8 };

  const geometry = useMemo(() => {
    if (data.length === 0 || width <= 0) {
      return null;
    }

    const plotWidth = Math.max(width - padding.left - padding.right, 1);
    const plotHeight = Math.max(height - padding.top - padding.bottom, 1);

    const xs = data.map((point) => point.x);
    const ys = data.map((point) => point.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    let minY = Math.min(...ys, showZeroLine ? 0 : Math.min(...ys));
    let maxY = Math.max(...ys, showZeroLine ? 0 : Math.max(...ys));

    if (minY === maxY) {
      // A flat series still deserves a sensible vertical window.
      const pad = Math.max(Math.abs(minY) * 0.1, 1);
      minY -= pad;
      maxY += pad;
    }

    const scaleX = (value: number) =>
      padding.left + (maxX === minX ? plotWidth / 2 : ((value - minX) / (maxX - minX)) * plotWidth);
    const scaleY = (value: number) =>
      padding.top + plotHeight - ((value - minY) / (maxY - minY)) * plotHeight;

    const points = data.map((point) => ({ ...point, cx: scaleX(point.x), cy: scaleY(point.y) }));

    const linePath = points
      .map((point, index) => `${index === 0 ? 'M' : 'L'}${point.cx.toFixed(2)},${point.cy.toFixed(2)}`)
      .join(' ');

    const baseline = padding.top + plotHeight;
    const first = points[0];
    const last = points[points.length - 1];
    const areaPath =
      first && last
        ? `${linePath} L${last.cx.toFixed(2)},${baseline.toFixed(2)} L${first.cx.toFixed(2)},${baseline.toFixed(2)} Z`
        : '';

    return { points, linePath, areaPath, scaleY, zeroY: scaleY(0), minY, maxY };
  }, [data, width, height, showZeroLine, padding.left, padding.right, padding.top, padding.bottom]);

  const handleTouch = useCallback(
    (event: GestureResponderEvent) => {
      if (!geometry || geometry.points.length === 0) return;
      const x = event.nativeEvent.locationX;
      let nearest = 0;
      let bestDistance = Infinity;
      geometry.points.forEach((point, index) => {
        const distance = Math.abs(point.cx - x);
        if (distance < bestDistance) {
          bestDistance = distance;
          nearest = index;
        }
      });
      setActiveIndex(nearest);
    },
    [geometry],
  );

  const styles = useMemo(
    () =>
      StyleSheet.create({
        wrapper: { gap: theme.spacing(2) },
        readout: { flexDirection: 'row', alignItems: 'baseline', gap: theme.spacing(2) },
        plot: { height, justifyContent: 'center' },
        empty: {
          height,
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: theme.radius.md,
          backgroundColor: theme.colors.surfaceAlt,
        },
      }),
    [theme, height],
  );

  const active = activeIndex !== null ? geometry?.points[activeIndex] : undefined;
  const latest = geometry?.points[geometry.points.length - 1];
  const shown = active ?? latest;

  if (data.length === 0) {
    return (
      <View style={styles.empty} testID={testID}>
        <Text tone="muted" variant="caption">
          {emptyMessage}
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.wrapper} testID={testID}>
      {shown ? (
        <View style={styles.readout}>
          <Text variant="heading" color={stroke}>
            {formatValue(shown.y)}
          </Text>
          <Text variant="caption" tone="muted">
            {formatLabel ? formatLabel(shown) : (shown.label ?? '')}
          </Text>
        </View>
      ) : null}

      <View
        style={styles.plot}
        onLayout={(event) => setWidth(event.nativeEvent.layout.width)}
        onStartShouldSetResponder={() => true}
        onMoveShouldSetResponder={() => true}
        onResponderGrant={handleTouch}
        onResponderMove={handleTouch}
        onResponderRelease={() => setActiveIndex(null)}
        onResponderTerminate={() => setActiveIndex(null)}
      >
        {geometry && width > 0 ? (
          <Svg width={width} height={height}>
            <Defs>
              <LinearGradient id="lineChartFill" x1="0" y1="0" x2="0" y2="1">
                <Stop offset="0" stopColor={stroke} stopOpacity={0.28} />
                <Stop offset="1" stopColor={stroke} stopOpacity={0.02} />
              </LinearGradient>
            </Defs>

            {showZeroLine &&
            geometry.minY <= 0 &&
            geometry.maxY >= 0 ? (
              <Line
                x1={0}
                y1={geometry.zeroY}
                x2={width}
                y2={geometry.zeroY}
                stroke={theme.colors.chartGrid}
                strokeWidth={1}
                strokeDasharray="4 4"
              />
            ) : null}

            {filled && geometry.areaPath ? (
              <Path d={geometry.areaPath} fill="url(#lineChartFill)" />
            ) : null}

            <Path
              d={geometry.linePath}
              stroke={stroke}
              strokeWidth={2.5}
              strokeLinejoin="round"
              strokeLinecap="round"
              fill="none"
            />

            {active ? (
              <>
                <Line
                  x1={active.cx}
                  y1={0}
                  x2={active.cx}
                  y2={height}
                  stroke={theme.colors.chartGrid}
                  strokeWidth={1}
                />
                <Circle
                  cx={active.cx}
                  cy={active.cy}
                  r={5}
                  fill={theme.colors.surface}
                  stroke={stroke}
                  strokeWidth={2.5}
                />
              </>
            ) : latest ? (
              <Circle cx={latest.cx} cy={latest.cy} r={4} fill={stroke} />
            ) : null}
          </Svg>
        ) : null}
      </View>
    </View>
  );
}
