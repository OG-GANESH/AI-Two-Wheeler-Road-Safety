// ============================================================
// AI-BASED RIDER MONITORING SYSTEM
// ESP32 BUZZER & LED WARNING CONTROLLER
// ============================================================

#define BUZZER_PIN    25
#define LED_YELLOW_PIN 27  // Warning Indicator
#define LED_RED_PIN    32  // Critical Indicator (DROWSY ONLY)

String command = "";
String currentStatus = "ALERT";

unsigned long previousMillis = 0;
int patternStep = 0;

// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(115200);

  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_YELLOW_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);

  // Initial state: Everything OFF
  noTone(BUZZER_PIN);
  digitalWrite(LED_YELLOW_PIN, LOW);
  digitalWrite(LED_RED_PIN, LOW);

  Serial.println();
  Serial.println("================================");
  Serial.println(" Rider Safety ESP32 Controller");
  Serial.println(" Buzzer & LEDs Ready");
  Serial.println("================================");
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {
  // Receive command from Python
  if (Serial.available()) {
    command = Serial.readStringUntil('\n');
    command.trim();

    if (command.length() > 0) {
      currentStatus = command;

      // Reset pattern timer and audio state on status update
      patternStep = 0;
      previousMillis = millis();
      noTone(BUZZER_PIN);

      Serial.print("[ESP32] Received: ");
      Serial.println(currentStatus);
    }
  }

  // Handle Warnings & Indicators
  if (currentStatus == "ALERT") {
    setLeds(LOW, LOW);
    alertPattern();
  }
  else if (currentStatus == "NO_FACE") {
    setLeds(LOW, LOW);
    noFacePattern();
  }
  else if (currentStatus == "DISTRACTED") {
    setLeds(HIGH, LOW); // Yellow Warning
    distractedPattern();
  }
  else if (currentStatus == "PHONE") {
    setLeds(HIGH, LOW); // Yellow Warning
    phonePattern();
  }
  else if (currentStatus == "EYES_CLOSED") {
    setLeds(HIGH, LOW); // Yellow Warning
    eyesClosedPattern();
  }
  else if (currentStatus == "DROWSY") {
    setLeds(LOW, HIGH); // RED Critical ONLY
    drowsyPattern();
  }
  else {
    // Unknown command fallback
    setLeds(LOW, LOW);
    noTone(BUZZER_PIN);
  }
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================

void setLeds(bool yellowState, bool redState) {
  digitalWrite(LED_YELLOW_PIN, yellowState);
  digitalWrite(LED_RED_PIN, redState);
}

void alertPattern() {
  noTone(BUZZER_PIN);
}

void noFacePattern() {
  noTone(BUZZER_PIN);
}

// ============================================================
// BUZZER PATTERNS
// ============================================================

void eyesClosedPattern() {
  unsigned long currentMillis = millis();

  if (patternStep == 0) {
    tone(BUZZER_PIN, 5000);
    if (currentMillis - previousMillis >= 180) {
      noTone(BUZZER_PIN);
      previousMillis = currentMillis;
      patternStep = 1;
    }
  } else if (patternStep == 1) {
    if (currentMillis - previousMillis >= 1000) {
      previousMillis = currentMillis;
      patternStep = 0;
    }
  }
}

void drowsyPattern() {
  unsigned long currentMillis = millis();

  if (patternStep == 0) {
    tone(BUZZER_PIN, 2500);
    if (currentMillis - previousMillis >= 180) {
      noTone(BUZZER_PIN);
      previousMillis = currentMillis;
      patternStep = 1;
    }
  } else if (patternStep == 1) {
    if (currentMillis - previousMillis >= 120) {
      tone(BUZZER_PIN, 2500);
      previousMillis = currentMillis;
      patternStep = 2;
    }
  } else if (patternStep == 2) {
    if (currentMillis - previousMillis >= 180) {
      noTone(BUZZER_PIN);
      previousMillis = currentMillis;
      patternStep = 3;
    }
  } else if (patternStep == 3) {
    if (currentMillis - previousMillis >= 120) {
      tone(BUZZER_PIN, 2500);
      previousMillis = currentMillis;
      patternStep = 4;
    }
  } else if (patternStep == 4) {
    if (currentMillis - previousMillis >= 180) {
      noTone(BUZZER_PIN);
      previousMillis = currentMillis;
      patternStep = 5;
    }
  } else if (patternStep == 5) {
    if (currentMillis - previousMillis >= 700) {
      previousMillis = currentMillis;
      patternStep = 0;
    }
  }
}

void distractedPattern() {
  unsigned long currentMillis = millis();

  if (patternStep == 0) {
    tone(BUZZER_PIN, 2000);
    if (currentMillis - previousMillis >= 250) {
      noTone(BUZZER_PIN);
      previousMillis = currentMillis;
      patternStep = 1;
    }
  } else if (patternStep == 1) {
    if (currentMillis - previousMillis >= 500) {
      previousMillis = currentMillis;
      patternStep = 0;
    }
  }
}

void phonePattern() {
  unsigned long currentMillis = millis();

  if (patternStep == 0) {
    tone(BUZZER_PIN, 1500);
    if (currentMillis - previousMillis >= 150) {
      noTone(BUZZER_PIN);
      previousMillis = currentMillis;
      patternStep = 1;
    }
  } else if (patternStep == 1) {
    if (currentMillis - previousMillis >= 150) {
      tone(BUZZER_PIN, 1500);
      previousMillis = currentMillis;
      patternStep = 2;
    }
  } else if (patternStep == 2) {
    if (currentMillis - previousMillis >= 150) {
      noTone(BUZZER_PIN);
      previousMillis = currentMillis;
      patternStep = 3;
    }
  } else if (patternStep == 3) {
    if (currentMillis - previousMillis >= 1000) {
      previousMillis = currentMillis;
      patternStep = 0;
    }
  }
}