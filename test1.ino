// =============================================================================
// ELEC3204 Project — Adaptive Cruise Control Motor Drive (FINAL SMOOTH VERSION)
// Fixes:
// - Smoothed measured speed (low-pass filter)
// - Increased speed measurement resolution
// - Removed hard stop reset jitter
// - Clean real-time plotting
// =============================================================================

#define ENC_A       2
#define ENC_B       3
#define IN1         4
#define IN2         5
#define TRIG_PIN    6
#define ECHO_PIN    7
#define BTN_PIN     8
#define PWM_PIN     9
#define LED_PIN     10

#define FORWARD  true
#define REVERSE  false

bool closedLoopMode = false;

// ── Encoder ───────────────────────────────────────────────────────────────────
volatile long encoderCount = 0;
volatile bool lastA        = false;

// ── Speed measurement ─────────────────────────────────────────────────────────
#define SPEED_INTERVAL  100   // 🔥 increased for smoother measurement

long  prevCount = 0;
long  prevTime  = 0;

float measuredSpeed = 0;
float filteredSpeed = 0;
float speedAlpha    = 0.7;   // smoothing factor

// ── PI Controller ─────────────────────────────────────────────────────────────
float targetSpeed = 0.0;

float Kp = 1.5;
float Ki = 0.4;

float integral = 0;

// ── ACC parameters ────────────────────────────────────────────────────────────
#define SAFE_DIST_CM   15.0
#define SLOW_DIST_CM   35.0
#define CRUISE_SPEED  200.0
#define MIN_SPEED      20.0

#define ULTRA_INTERVAL  80
long lastUltraTime = 0;

// ── Distance smoothing ────────────────────────────────────────────────────────
float filteredDist = 100.0;
float alpha = 0.7;

// ── PWM ramp ──────────────────────────────────────────────────────────────────
int lastPwm = 0;
int maxStep = 10;

// ── Open-loop stages ──────────────────────────────────────────────────────────
struct Stage { bool fwd; int pwm; long dur; };

static const Stage stages[] = {
  { FORWARD, 100, 2000 },
  { FORWARD, 200, 2000 },
  { FORWARD,   0,  500 },
  { REVERSE, 150, 2000 },
  { REVERSE,   0,  500 },
};

const int NUM_STAGES = 5;
static int  olState = 0;
static long olTimer = 0;

// =============================================================================
// Helpers
// =============================================================================
void setMotor(bool forward, int pwmVal) {
  pwmVal = constrain(pwmVal, 0, 255);
  digitalWrite(IN1, forward ? HIGH : LOW);
  digitalWrite(IN2, forward ? LOW  : HIGH);
  analogWrite(PWM_PIN, pwmVal);
}

void stopMotor() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(PWM_PIN, 0);
}

// ── Ultrasonic ────────────────────────────────────────────────────────────────
float readDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 15000);

  if (duration == 0) return -1;

  return duration * 0.0343 / 2.0;
}

// ── ACC mapping ───────────────────────────────────────────────────────────────
float accSetpoint(float distCM) {
  if (distCM <= SAFE_DIST_CM) {
    return 0;
  } else if (distCM <= SLOW_DIST_CM) {
    float ratio = (distCM - SAFE_DIST_CM) / (SLOW_DIST_CM - SAFE_DIST_CM);
    return ratio * CRUISE_SPEED;
  } else {
    return CRUISE_SPEED;
  }
}

// ── Encoder ISR ───────────────────────────────────────────────────────────────
void encoderISR() {
  bool A = digitalRead(ENC_A);
  bool B = digitalRead(ENC_B);

  if (A != lastA) {
    encoderCount += (A == B) ? -1 : 1;
    lastA = A;
  }
}

// ── Button ────────────────────────────────────────────────────────────────────
bool buttonPressed() {
  static bool lastReading = HIGH;
  bool reading = digitalRead(BTN_PIN);

  bool pressed = (lastReading == HIGH && reading == LOW);

  lastReading = reading;
  return pressed;
}

// =============================================================================
void setup() {
  Serial.begin(115200);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(PWM_PIN, OUTPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BTN_PIN, INPUT_PULLUP);
  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENC_A), encoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B), encoderISR, CHANGE);

  lastA    = digitalRead(ENC_A);
  prevTime = millis();
  olTimer  = millis();

  Serial.println("=== FINAL SMOOTH ACC MOTOR READY ===");
}

// =============================================================================
void loop() {
  long now = millis();

  // ── Mode toggle ────────────────────────────────────────────────────────────
  if (buttonPressed()) {
    closedLoopMode = !closedLoopMode;
    stopMotor();
    olState = 0;
    olTimer = now;
    integral = 0;

    digitalWrite(LED_PIN, closedLoopMode ? HIGH : LOW);
  }

  // ── Speed measurement (SMOOTHED) ───────────────────────────────────────────
  if (now - prevTime >= SPEED_INTERVAL) {

    long dt    = now - prevTime;
    long delta = encoderCount - prevCount;

    float rawSpeed = (float)delta / dt * 1000.0;

    // 🔥 Low-pass filter
    filteredSpeed = speedAlpha * filteredSpeed + (1.0 - speedAlpha) * rawSpeed;

    measuredSpeed = filteredSpeed;

    prevCount = encoderCount;
    prevTime  = now;
  }

  // ── Mode select ───────────────────────────────────────────────────────────
  if (!closedLoopMode) {
    runOpenLoop(now);
  } else {
    runClosedLoop(now);
  }

  delay(10);
}

// =============================================================================
// Open-loop
// =============================================================================
void runOpenLoop(long now) {
  if (now - olTimer >= stages[olState].dur) {
    olState = (olState + 1) % NUM_STAGES;
    olTimer = now;
  }

  setMotor(stages[olState].fwd, stages[olState].pwm);
}

// =============================================================================
// Closed-loop (SMOOTH)
// =============================================================================
void runClosedLoop(long now) {

  // ── Ultrasonic + smoothing ────────────────────────────────────────────────
  if (now - lastUltraTime >= ULTRA_INTERVAL) {
    float newDist = readDistanceCM();

    if (newDist > 0 && newDist < 200) {
      filteredDist = alpha * filteredDist + (1.0 - alpha) * newDist;
    }

    targetSpeed   = accSetpoint(filteredDist);
    lastUltraTime = now;
  }

  // ── PI Control ────────────────────────────────────────────────────────────
  float dt_s  = SPEED_INTERVAL / 1000.0;
  float speed = abs(measuredSpeed);

  float error = targetSpeed - speed;

  integral += error * dt_s;
  integral  = constrain(integral, -200, 200);

  float output = Kp * error + Ki * integral;

  int pwmOut = constrain((int)output, 0, 255);

  // ── PWM ramp limiting ─────────────────────────────────────────────────────
  pwmOut  = constrain(pwmOut, lastPwm - maxStep, lastPwm + maxStep);
  lastPwm = pwmOut;

  // ── Apply motor control ───────────────────────────────────────────────────
  if (targetSpeed < MIN_SPEED) {
    setMotor(FORWARD, 0);   // 🔥 no reset → smoother behavior
  } else {
    setMotor(FORWARD, pwmOut);
  }

  // ── Serial Plotter Output (clean format) ──────────────────────────────────
  Serial.print(filteredDist); Serial.print(" ");
  Serial.print(targetSpeed);  Serial.print(" ");
  Serial.print(speed);        Serial.print(" ");
  Serial.println(pwmOut);
}
