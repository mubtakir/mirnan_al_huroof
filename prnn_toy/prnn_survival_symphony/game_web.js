class Enemy {
    constructor(id) {
        this.id = id;
        this.distance = 1.3;
        this.angle = Math.random() * 2 * Math.PI;
        const freqOptions = [0.2, 0.5, 0.8];
        this.frequency = freqOptions[Math.floor(Math.random() * 3)];
        this.phase = Math.random() * 2 * Math.PI;
        this.health = 100;
        this.color = this.frequency === 0.2 ? 'red' : (this.frequency === 0.5 ? 'yellow' : 'magenta');
        this.morphTimer = 0;
        this.isChameleon = false;
        this.radius = 8;
    }
}

class Player {
    constructor() {
        this.frequency = 0.3;
        this.phase = 0.0;
        this.shieldActive = false;
        this.shieldEnergy = 100.0;
        this.health = 100.0;
        this.score = 0;
        this.autopilot = true;
    }
}

class Particle {
    constructor(x, y, color) {
        this.x = x;
        this.y = y;
        this.vx = (Math.random() - 0.5) * 4;
        this.vy = (Math.random() - 0.5) * 2;
        this.life = 8 + Math.floor(Math.random() * 8);
        this.maxLife = this.life;
        this.color = color;
    }
}

class GameState {
    constructor() {
        this.enemies = [];
        this.player = new Player();
        this.particles = [];
        this.idCounter = 1;
        this.spawnTimer = 0;
        this.spawnInterval = 45;
        this.waveDifficulty = 1;
        this.reportActive = false;
        this.creditsActive = false;
    }
}

function spawnEnemy(state) {
    const enemy = new Enemy(state.idCounter++);
    if (state.waveDifficulty >= 3 && Math.random() < 0.1) {
        enemy.frequency = 0.35;
        enemy.color = 'green';
        enemy.isChameleon = true;
    }
    state.enemies.push(enemy);
}

function spawnExplosion(state, ex, ey, color) {
    for (let i = 0; i < 12; i++) {
        state.particles.push(new Particle(ex, ey, color));
    }
}

function updateGame(state, dt) {
    const player = state.player;

    player.phase = (player.phase + 2 * Math.PI * player.frequency * dt) % (2 * Math.PI);

    if (player.shieldActive) {
        player.shieldEnergy = Math.max(0, player.shieldEnergy - 25 * dt);
        if (player.shieldEnergy <= 0) player.shieldActive = false;
    } else {
        player.shieldEnergy = Math.min(100, player.shieldEnergy + 10 * dt);
    }

    state.enemies = state.enemies.filter(enemy => {
        let speed = Math.min(0.35, 0.08 + state.waveDifficulty * 0.01);
        const freqDiff = Math.abs(player.frequency - enemy.frequency);
        if (freqDiff > 0.32) speed *= 1.5;

        enemy.distance -= speed * dt;
        enemy.phase = (enemy.phase + 2 * Math.PI * enemy.frequency * dt) % (2 * Math.PI);

        if (enemy.isChameleon) {
            enemy.morphTimer++;
            if (enemy.morphTimer >= 180) {
                enemy.morphTimer = 0;
                const others = [0.2, 0.5, 0.8].filter(f => f !== enemy.frequency);
                enemy.frequency = others[Math.floor(Math.random() * others.length)];
                enemy.color = enemy.frequency === 0.2 ? 'red' : (enemy.frequency === 0.5 ? 'yellow' : 'magenta');
            }
        }

        if (enemy.distance <= 0.18) {
            let damage = 15;
            const fd = Math.abs(player.frequency - enemy.frequency);
            if (player.shieldActive && fd <= 0.1) damage = 2;
            else if (player.shieldActive) damage = 7;
            player.health = Math.max(0, player.health - damage);
            spawnExplosion(state, 0, 0, 'red');
            return false;
        }

        if (player.shieldActive && freqDiff <= 0.08) {
            const interference = Math.cos(player.phase - enemy.phase);
            let damageRate = 80 * (1 - freqDiff / 0.08);
            if (interference > 0) damageRate *= (1 + 1 * interference);
            else damageRate *= (1 + 0.3 * interference);
            enemy.health -= damageRate * dt;
            if (Math.random() < 0.4) {
                const ex = enemy.distance * Math.cos(enemy.angle) * 200;
                const ey = enemy.distance * Math.sin(enemy.angle) * 200;
                spawnExplosion(state, ex + 350, ey + 250, enemy.color);
            }
        }

        if (enemy.health <= 0) {
            player.score += 10;
            const ex = enemy.distance * Math.cos(enemy.angle) * 200;
            const ey = enemy.distance * Math.sin(enemy.angle) * 200;
            spawnExplosion(state, ex + 350, ey + 250, enemy.color);
            if (player.score % 50 === 0) {
                state.waveDifficulty++;
                state.spawnInterval = Math.max(20, state.spawnInterval - 5);
            }
            return false;
        }
        return true;
    });

    state.particles = state.particles.filter(p => {
        p.x += p.vx; p.y += p.vy;
        p.life--;
        return p.life > 0;
    });

    state.spawnTimer++;
    if (state.spawnTimer >= state.spawnInterval) {
        state.spawnTimer = 0;
        spawnEnemy(state);
    }
}
