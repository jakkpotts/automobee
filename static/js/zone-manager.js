class ZoneManager {
    constructor() {
        this.zones = new Map();
        this.activeZones = new Set();
        this.processingLimits = {
            activePerZone: 35,  // Active processing limit per zone
            reservePerZone: 12, // Reserve capacity per zone
            maxZones: 5        // Maximum concurrent active zones
        };
        this.processingQueues = {
            high: new Map(),   // Priority queue for active zones
            low: new Map()     // Background queue for inactive zones
        };
    }

    createZone(zoneId, center, radius) {
        this.zones.set(zoneId, {
            id: zoneId,
            center,
            radius,
            cameras: new Set(),
            activeStreams: 0,
            queuedStreams: [],
            stats: {
                totalDetections: 0,
                targetDetections: 0,
                nonTargetDetections: 0,
                avgProcessingTime: 0
            }
        });
    }

    assignCameraToZone(cameraId, location) {
        const nearestZone = this.findNearestZone(location);
        if (nearestZone) {
            nearestZone.cameras.add(cameraId);
            this.rebalanceZoneProcessing(nearestZone.id);
        }
    }

    activateZone(zoneId) {
        if (this.activeZones.size >= this.processingLimits.maxZones) {
            const leastActiveZone = this.findLeastActiveZone();
            this.deactivateZone(leastActiveZone.id);
        }
        
        this.activeZones.add(zoneId);
        this.promoteQueuedStreams(zoneId);
    }

    async promoteQueuedStreams(zoneId) {
        const zone = this.zones.get(zoneId);
        const availableSlots = this.processingLimits.activePerZone - zone.activeStreams;
        
        for (let i = 0; i < availableSlots && zone.queuedStreams.length > 0; i++) {
            const camera = zone.queuedStreams.shift();
            await this.activateStream(camera, zoneId);
        }
    }

    async activateStream(cameraId, zoneId) {
        const zone = this.zones.get(zoneId);
        if (zone.activeStreams < this.processingLimits.activePerZone) {
            zone.activeStreams++;
            this.processingQueues.high.set(cameraId, {
                zoneId,
                priority: this.calculatePriority(cameraId, zoneId)
            });
            return true;
        }
        return false;
    }

    calculatePriority(cameraId, zoneId) {
        const zone = this.zones.get(zoneId);
        return {
            zoneActive: this.activeZones.has(zoneId),
            streamHealth: this.getStreamHealth(cameraId),
            detectionRate: zone.stats.targetDetections / zone.stats.totalDetections,
            processingEfficiency: zone.stats.avgProcessingTime
        };
    }

    async processZoneQueues() {
        for (const zoneId of this.activeZones) {
            const zone = this.zones.get(zoneId);
            
            // Process high-priority queue first
            for (const [cameraId, config] of this.processingQueues.high) {
                if (config.zoneId === zoneId) {
                    await this.processStream(cameraId, true);
                }
            }

            // Process low-priority queue if capacity available
            if (zone.activeStreams < this.processingLimits.activePerZone) {
                for (const [cameraId, config] of this.processingQueues.low) {
                    if (config.zoneId === zoneId) {
                        await this.processStream(cameraId, false);
                    }
                }
            }
        }
    }
} 