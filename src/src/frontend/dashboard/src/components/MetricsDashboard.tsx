import { Card, Title, Text, AreaChart, DonutChart, Flex, Grid } from '@tremor/react';
import type { SystemMetrics } from '../types';

interface MetricsDashboardProps {
  metrics: SystemMetrics | null;
}

const formatNumber = (num: number) => {
  return new Intl.NumberFormat('en-US').format(num);
};

export function MetricsDashboard({ metrics }: MetricsDashboardProps) {
  if (!metrics) {
    return (
      <div className="h-full flex items-center justify-center">
        <Text>Loading metrics...</Text>
      </div>
    );
  }

  const resourceUsage = [
    { name: 'CPU', value: metrics.cpuUsage },
    { name: 'Memory', value: metrics.memoryUsage },
    { name: 'GPU', value: metrics.gpuUtilization },
  ];

  return (
    <div className="space-y-6">
      <Title>System Metrics</Title>
      
      {/* Key Metrics */}
      <Grid numItems={2} className="gap-4">
        <Card decoration="top" decorationColor="blue">
          <Title>Total Detections</Title>
          <Text className="mt-2 text-2xl font-bold">
            {formatNumber(metrics.totalDetections)}
          </Text>
        </Card>
        <Card decoration="top" decorationColor="green">
          <Title>Active Cameras</Title>
          <Text className="mt-2 text-2xl font-bold">
            {metrics.activeCameras}
          </Text>
        </Card>
      </Grid>

      {/* Detection Rate Chart */}
      <Card>
        <Title>Detection Rate</Title>
        <Text className="mt-2">Detections per minute</Text>
        <AreaChart
          className="mt-4 h-28"
          data={[
            { time: '1m ago', value: metrics.detectionRate },
            { time: 'now', value: metrics.detectionRate * 1.1 }, // Simulated variation
          ]}
          index="time"
          categories={['value']}
          colors={['blue']}
          showXAxis={false}
          showGridLines={false}
          showAnimation
        />
      </Card>

      {/* Resource Usage */}
      <Card>
        <Title>Resource Usage</Title>
        <DonutChart
          className="mt-4 h-32"
          data={resourceUsage}
          category="value"
          index="name"
          colors={['blue', 'cyan', 'indigo']}
          showAnimation
        />
        <Flex className="mt-4 space-x-2" justifyContent="center">
          {resourceUsage.map(({ name, value }) => (
            <Text key={name} className="text-sm">
              {name}: {value.toFixed(1)}%
            </Text>
          ))}
        </Flex>
      </Card>
    </div>
  );
} 