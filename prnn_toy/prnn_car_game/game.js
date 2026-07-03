/**
 * Cyberpunk Car Simulation and Track Logic
 * Handles raycasting, physics, track generation, and PRNN visualization
 */

class Car {
    constructor(x, y, angle) {
        this.startX = x;
        this.startY = y;
        this.startAngle = angle;
        
        this.x = x;
        this.y = y;
        this.angle = angle;
        this.speed = 3.5;
        this.radius = 12;
        
        // Raycasting sensor setup: angles relative to car orientation
        this.sensorAngles = [-Math.PI / 4, 0, Math.PI / 4]; // -45, 0, 45 degrees
        this.sensorRange = 150;
        this.sensorDistances = [1.0, 1.0, 1.0]; // Normalized [0, 1] (1 is clear, 0 is collision)
        this.sensorIntersections = [null, null, null];
    }

    reset() {
        this.x = this.startX;
        this.y = this.startY;
        this.angle = this.startAngle;
        this.sensorDistances = [1.0, 1.0, 1.0];
        this.sensorIntersections = [null, null, null];
    }

    update(steeringInput) {
        // Limit steering angular velocity
        const maxSteer = 0.08;
        const steer = steeringInput * maxSteer;
        this.angle += steer;
        
        // Move car forward
        this.x += Math.cos(this.angle) * this.speed;
        this.y += Math.sin(this.angle) * this.speed;
    }

    castRays(walls) {
        for (let i = 0; i < this.sensorAngles.length; i++) {
            const rayAngle = this.angle + this.sensorAngles[i];
            const rayEnd = {
                x: this.x + Math.cos(rayAngle) * this.sensorRange,
                y: this.y + Math.sin(rayAngle) * this.sensorRange
            };
            
            let closestIntersect = null;
            let minDistance = this.sensorRange;
            
            for (const wall of walls) {
                const intersect = this.getIntersection(
                    { x: this.x, y: this.y },
                    rayEnd,
                    wall.p1,
                    wall.p2
                );
                
                if (intersect) {
                    const dist = Math.hypot(intersect.x - this.x, intersect.y - this.y);
                    if (dist < minDistance) {
                        minDistance = dist;
                        closestIntersect = intersect;
                    }
                }
            }
            
            this.sensorIntersections[i] = closestIntersect;
            this.sensorDistances[i] = minDistance / this.sensorRange;
        }
    }

    getIntersection(A, B, C, D) {
        const tTop = (D.x - C.x) * (A.y - C.y) - (D.y - C.y) * (A.x - C.x);
        const uTop = (C.y - A.y) * (A.x - B.x) - (C.x - A.x) * (A.y - B.y);
        const bottom = (D.y - C.y) * (B.x - A.x) - (D.x - C.x) * (B.y - A.y);
        
        if (bottom !== 0) {
            const t = tTop / bottom;
            const u = uTop / bottom;
            if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
                return {
                    x: A.x + t * (B.x - A.x),
                    y: A.y + t * (B.y - A.y)
                };
            }
        }
        return null;
    }

    draw(ctx) {
        // Draw sensor rays
        for (let i = 0; i < this.sensorAngles.length; i++) {
            const rayAngle = this.angle + this.sensorAngles[i];
            const endX = this.sensorIntersections[i] ? this.sensorIntersections[i].x : this.x + Math.cos(rayAngle) * this.sensorRange;
            const endY = this.sensorIntersections[i] ? this.sensorIntersections[i].y : this.y + Math.sin(rayAngle) * this.sensorRange;
            
            ctx.beginPath();
            ctx.moveTo(this.x, this.y);
            ctx.lineTo(endX, endY);
            
            // Fade color from green (safe) to glowing red (danger)
            const dist = this.sensorDistances[i];
            ctx.strokeStyle = dist < 0.4 ? `rgba(239, 68, 68, ${0.4 + (1 - dist) * 0.6})` : `rgba(16, 185, 129, ${0.1 + (1 - dist) * 0.4})`;
            ctx.lineWidth = dist < 0.4 ? 2.5 : 1.5;
            ctx.stroke();
            
            // Draw sensor contact points
            if (this.sensorIntersections[i]) {
                ctx.beginPath();
                ctx.arc(endX, endY, 5, 0, 2 * Math.PI);
                ctx.fillStyle = dist < 0.4 ? '#ef4444' : '#10b981';
                ctx.shadowColor = dist < 0.4 ? '#ef4444' : '#10b981';
                ctx.shadowBlur = 8;
                ctx.fill();
                ctx.shadowBlur = 0;
            }
        }
        
        // Draw car chassis
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(this.angle);
        
        // Glow effect
        ctx.shadowColor = '#3b82f6';
        ctx.shadowBlur = 12;
        
        ctx.beginPath();
        // Cyberpunk arrow-like vehicle
        ctx.moveTo(18, 0);
        ctx.lineTo(-10, -12);
        ctx.lineTo(-6, -6);
        ctx.lineTo(-6, 6);
        ctx.lineTo(-10, 12);
        ctx.closePath();
        ctx.fillStyle = '#3b82f6';
        ctx.fill();
        
        ctx.strokeStyle = '#60a5fa';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Cabin cockpit glass
        ctx.beginPath();
        ctx.moveTo(6, 0);
        ctx.lineTo(-4, -5);
        ctx.lineTo(-4, 5);
        ctx.closePath();
        ctx.fillStyle = '#a78bfa';
        ctx.fill();
        
        ctx.restore();
    }
}

class Track {
    constructor(type) {
        this.walls = [];
        this.generate(type);
    }

    generate(type) {
        this.walls = [];
        const width = 800;
        const height = 500;
        
        if (type === 'oval') {
            // Oval Track
            const cx = width / 2;
            const cy = height / 2;
            const segments = 32;
            
            const outerRX = 360;
            const outerRY = 210;
            const innerRX = 220;
            const innerRY = 90;
            
            const outerPoints = [];
            const innerPoints = [];
            
            for (let i = 0; i < segments; i++) {
                const theta = (i / segments) * 2 * Math.PI;
                outerPoints.push({
                    x: cx + Math.cos(theta) * outerRX,
                    y: cy + Math.sin(theta) * outerRY
                });
                innerPoints.push({
                    x: cx + Math.cos(theta) * innerRX,
                    y: cy + Math.sin(theta) * innerRY
                });
            }
            
            this.buildLoopWalls(outerPoints);
            this.buildLoopWalls(innerPoints);
            
        } else if (type === 'scurve') {
            // S-Curve Track (Winding Road)
            // Left outer bounds
            const leftOuter = [
                {x: 50, y: 450}, {x: 50, y: 150}, {x: 150, y: 50},
                {x: 350, y: 50}, {x: 450, y: 150}, {x: 450, y: 350},
                {x: 550, y: 450}, {x: 750, y: 450}, {x: 750, y: 150},
                {x: 650, y: 50}, {x: 500, y: 50}, {x: 350, y: 150},
                {x: 250, y: 350}, {x: 150, y: 450}
            ];
            
            // Corresponding inner bounds (offset)
            const leftInner = [
                {x: 150, y: 350}, {x: 150, y: 200}, {x: 200, y: 150},
                {x: 300, y: 150}, {x: 350, y: 200}, {x: 350, y: 300},
                {x: 450, y: 350}, {x: 650, y: 350}, {x: 650, y: 250},
                {x: 600, y: 180}, {x: 520, y: 150}, {x: 420, y: 250},
                {x: 320, y: 350}, {x: 200, y: 380}
            ];
            
            this.buildLoopWalls(leftOuter);
            this.buildLoopWalls(leftInner);
            
        } else if (type === 'obstacles') {
            // Circular track with obstacles in the path
            const cx = width / 2;
            const cy = height / 2;
            const segments = 24;
            
            const outerPoints = [];
            const innerPoints = [];
            
            for (let i = 0; i < segments; i++) {
                const theta = (i / segments) * 2 * Math.PI;
                outerPoints.push({ x: cx + Math.cos(theta) * 350, y: cy + Math.sin(theta) * 220 });
                innerPoints.push({ x: cx + Math.cos(theta) * 200, y: cy + Math.sin(theta) * 100 });
            }
            
            this.buildLoopWalls(outerPoints);
            this.buildLoopWalls(innerPoints);
            
            // Add specific obstacles in the middle of the track lanes
            // Obstacle 1: Box Top-Left
            this.addBoxObstacle(200, 120, 40, 40);
            // Obstacle 2: Box Bottom-Right
            this.addBoxObstacle(600, 340, 40, 40);
            // Obstacle 3: Barrier Right
            this.addBoxObstacle(620, 220, 60, 20);
            // Obstacle 4: Barrier Left
            this.addBoxObstacle(120, 260, 20, 60);
        }
    }

    buildLoopWalls(points) {
        for (let i = 0; i < points.length; i++) {
            const p1 = points[i];
            const p2 = points[(i + 1) % points.length];
            this.walls.push({ p1, p2 });
        }
    }

    addBoxObstacle(x, y, w, h) {
        const topL = { x: x, y: y };
        const topR = { x: x + w, y: y };
        const botL = { x: x, y: y + h };
        const botR = { x: x + w, y: y + h };
        
        this.walls.push({ p1: topL, p2: topR });
        this.walls.push({ p1: topR, p2: botR });
        this.walls.push({ p1: botR, p2: botL });
        this.walls.push({ p1: botL, p2: topL });
    }

    checkCollision(car) {
        for (const wall of this.walls) {
            const dist = this.getPointToSegmentDistance(
                { x: car.x, y: car.y },
                wall.p1,
                wall.p2
            );
            if (dist < car.radius) {
                return true; // Crash detected!
            }
        }
        return false;
    }

    getPointToSegmentDistance(pt, v, w) {
        const l2 = Math.pow(v.x - w.x, 2) + Math.pow(v.y - w.y, 2);
        if (l2 === 0) return Math.hypot(pt.x - v.x, pt.y - v.y);
        
        let t = ((pt.x - v.x) * (w.x - v.x) + (pt.y - v.y) * (w.y - v.y)) / l2;
        t = Math.max(0, Math.min(1, t));
        
        return Math.hypot(
            pt.x - (v.x + t * (w.x - v.x)),
            pt.y - (v.y + t * (w.y - v.y))
        );
    }

    draw(ctx) {
        ctx.beginPath();
        for (const wall of this.walls) {
            ctx.moveTo(wall.p1.x, wall.p1.y);
            ctx.lineTo(wall.p2.x, wall.p2.y);
        }
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.4)';
        ctx.lineWidth = 3;
        ctx.shadowColor = '#8b5cf6';
        ctx.shadowBlur = 6;
        ctx.stroke();
        ctx.shadowBlur = 0;
    }
}
