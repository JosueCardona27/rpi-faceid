#include <Servo.h>

#define PIN_SERVO     9
#define PIN_LED_VERDE 6
#define PIN_LED_ROJO  7

#define SERVO_ABIERTO  90
#define SERVO_CERRADO   0

#define TIEMPO_PUERTA_ABIERTA  5000
#define TIEMPO_LED_DENEGADO    3000
#define TIEMPO_PARPADEO         300

Servo servo;

bool puerta_abierta = false;
unsigned long t_apertura = 0;
bool denegando = false;
unsigned long t_denegado = 0;
unsigned long t_parpadeo = 0;
bool led_rojo_encendido = false;
String buffer_serial = "";

void setup() {
  Serial.begin(9600);
  pinMode(PIN_LED_VERDE, OUTPUT);
  pinMode(PIN_LED_ROJO,  OUTPUT);
  digitalWrite(PIN_LED_VERDE, LOW);
  digitalWrite(PIN_LED_ROJO,  LOW);
  servo.attach(PIN_SERVO);
  servo.write(SERVO_CERRADO);
  Serial.println("=== SISTEMA LISTO ===");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      buffer_serial.trim();
      procesar_comando(buffer_serial);
      buffer_serial = "";
    } else {
      buffer_serial += c;
    }
  }

  if (puerta_abierta && millis() - t_apertura >= TIEMPO_PUERTA_ABIERTA) {
    cerrar_puerta();
  }

  if (denegando) {
    if (millis() - t_parpadeo >= TIEMPO_PARPADEO) {
      t_parpadeo = millis();
      led_rojo_encendido = !led_rojo_encendido;
      digitalWrite(PIN_LED_ROJO, led_rojo_encendido);
    }
    if (millis() - t_denegado >= TIEMPO_LED_DENEGADO) {
      denegando = false;
      led_rojo_encendido = false;
      digitalWrite(PIN_LED_ROJO, LOW);
    }
  }
}

void procesar_comando(String cmd) {
  cmd.toLowerCase();
  Serial.print("Comando recibido: ");
  Serial.println(cmd);

  if (cmd.startsWith("acceso:")) {
    String nombre = cmd.substring(7);
    accion_acceso_permitido(nombre);
  }
  else if (cmd == "denegado") { accion_acceso_denegado(); }
  else if (cmd == "espera")   { accion_espera(); }
  else if (cmd == "abrir") {
    denegando = false;
    digitalWrite(PIN_LED_ROJO,  LOW);
    digitalWrite(PIN_LED_VERDE, HIGH);
    servo.write(SERVO_ABIERTO);
  }
  else if (cmd == "cerrar") {
    denegando = false;
    digitalWrite(PIN_LED_VERDE, LOW);
    servo.write(SERVO_CERRADO);
  }
}

void accion_acceso_permitido(String nombre) {
  Serial.println("ACCESO PERMITIDO: " + nombre);
  denegando = false;
  digitalWrite(PIN_LED_ROJO,  LOW);
  digitalWrite(PIN_LED_VERDE, HIGH);
  servo.write(SERVO_ABIERTO);
  puerta_abierta = true;
  t_apertura = millis();
}

void accion_acceso_denegado() {
  Serial.println("ACCESO DENEGADO");
  if (puerta_abierta) { cerrar_puerta(); }
  digitalWrite(PIN_LED_VERDE, LOW);
  denegando = true;
  t_denegado = millis();
  t_parpadeo = millis();
  led_rojo_encendido = true;
  digitalWrite(PIN_LED_ROJO, HIGH);
}

void accion_espera() {
  Serial.println("MODO ESPERA");
  denegando = false;
  digitalWrite(PIN_LED_VERDE, LOW);
  digitalWrite(PIN_LED_ROJO,  LOW);
  if (puerta_abierta) { cerrar_puerta(); }
}

void cerrar_puerta() {
  servo.write(SERVO_CERRADO);
  puerta_abierta = false;
  digitalWrite(PIN_LED_VERDE, LOW);
}
