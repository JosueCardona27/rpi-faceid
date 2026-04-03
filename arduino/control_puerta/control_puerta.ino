/*
 * ================================================================
 *  control_puerta.ino
 *  Sistema de Control de Acceso Facial — Arduino Nano
 * ================================================================
 *  HARDWARE:
 *    - Arduino Nano
 *    - Servomotor en pin D9  (señal)
 *    - LED VERDE  en pin D6  (acceso permitido)
 *    - LED ROJO   en pin D7  (acceso denegado)
 *    - Botón de apertura manual en pin D4
 *        → Un lado del botón a D4
 *        → El otro lado a GND
 *        (el pin usa INPUT_PULLUP interno, no necesita resistencia)
 *
 *  MODO PRUEBA (Serial Monitor 9600):
 *    acceso:Juan
 *    denegado
 *    espera
 *    abrir
 *    cerrar
 * ================================================================
 */

#include <Servo.h>

// ── Configuración de pines ────────────────────────────────────────
#define PIN_SERVO     9
#define PIN_LED_VERDE 6
#define PIN_LED_ROJO  7
#define PIN_BOTON     4   // botón de apertura manual

// ── Ángulos del servomotor ───────────────────────────────────────
#define SERVO_ABIERTO  90
#define SERVO_CERRADO   0

// ── Tiempos ──────────────────────────────────────────────────────
#define TIEMPO_PUERTA_ABIERTA  4000
#define TIEMPO_LED_DENEGADO    3000
#define TIEMPO_PARPADEO         300
#define DEBOUNCE_MS              50   // anti-rebote del botón

// ── Objetos ──────────────────────────────────────────────────────
Servo servo;

// ── Variables ────────────────────────────────────────────────────
bool puerta_abierta = false;
unsigned long t_apertura = 0;

bool denegando = false;
unsigned long t_denegado = 0;
unsigned long t_parpadeo = 0;
bool led_rojo_encendido = false;

String buffer_serial = "";

// ── Variables del botón (anti-rebote) ────────────────────────────
bool     boton_estado_anterior = HIGH;
bool     boton_estable         = HIGH;
unsigned long t_debounce       = 0;


// ════════════════════════════════════════════════════════════════
void setup() {

  Serial.begin(9600);

  pinMode(PIN_LED_VERDE, OUTPUT);
  pinMode(PIN_LED_ROJO,  OUTPUT);
  pinMode(PIN_BOTON,     INPUT_PULLUP);

  digitalWrite(PIN_LED_VERDE, LOW);
  digitalWrite(PIN_LED_ROJO,  LOW);

  servo.attach(PIN_SERVO);
  servo.write(SERVO_CERRADO);

  Serial.println("=== SISTEMA LISTO ===");
}


// ════════════════════════════════════════════════════════════════
void loop() {

  // ── Lectura Serial ──────────────────────────────────────────
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

  // ── Botón de apertura manual ────────────────────────────────
  leer_boton();

  // ── Auto cierre puerta ──────────────────────────────────────
  if (puerta_abierta && millis() - t_apertura >= TIEMPO_PUERTA_ABIERTA) {
    cerrar_puerta();
  }

  // ── Parpadeo LED rojo ───────────────────────────────────────
  if (denegando) {

    if (millis() - t_parpadeo >= TIEMPO_PARPADEO) {
      t_parpadeo = millis();
      led_rojo_encendido = !led_rojo_encendido;
      digitalWrite(PIN_LED_ROJO, led_rojo_encendido);
    }

    if (millis() - t_denegado >= TIEMPO_LED_DENEGADO) {
      denegando          = false;
      led_rojo_encendido = false;
      digitalWrite(PIN_LED_ROJO, LOW);
    }
  }
}


// ════════════════════════════════════════════════════════════════
//  BOTÓN CON ANTI-REBOTE
// ════════════════════════════════════════════════════════════════
void leer_boton() {
  bool lectura = digitalRead(PIN_BOTON);

  if (lectura != boton_estado_anterior) {
    t_debounce = millis();
    boton_estado_anterior = lectura;
  }

  if (millis() - t_debounce >= DEBOUNCE_MS) {
    if (lectura != boton_estable) {
      boton_estable = lectura;

      // Flanco de bajada → botón presionado
      if (boton_estable == LOW) {
        Serial.println("BOTON: apertura manual");
        accion_boton_manual();
      }
    }
  }
}


// ════════════════════════════════════════════════════════════════
//  APERTURA MANUAL POR BOTÓN
// ════════════════════════════════════════════════════════════════
void accion_boton_manual() {

  // Si ya está abierta, solo reiniciar el temporizador
  if (puerta_abierta) {
    t_apertura = millis();
    return;
  }

  denegando          = false;
  led_rojo_encendido = false;
  digitalWrite(PIN_LED_ROJO, LOW);

  digitalWrite(PIN_LED_VERDE, HIGH);
  servo.write(SERVO_ABIERTO);
  puerta_abierta = true;
  t_apertura     = millis();
}


// ════════════════════════════════════════════════════════════════
//  PROCESAMIENTO DE COMANDOS
// ════════════════════════════════════════════════════════════════
void procesar_comando(String cmd) {

  cmd.toLowerCase();

  Serial.print("Comando recibido: ");
  Serial.println(cmd);

  if (cmd.startsWith("acceso:")) {
    String nombre = cmd.substring(7);
    accion_acceso_permitido(nombre);
  }

  else if (cmd == "denegado") {
    accion_acceso_denegado();
  }

  else if (cmd == "espera") {
    accion_espera();
  }

  // ── PRUEBAS DIRECTAS ────────────────────────────────────────
  else if (cmd == "abrir") {
    Serial.println("Abriendo servo...");
    denegando = false;
    digitalWrite(PIN_LED_ROJO,  LOW);
    digitalWrite(PIN_LED_VERDE, HIGH);
    servo.write(SERVO_ABIERTO);
  }

  else if (cmd == "cerrar") {
    Serial.println("Cerrando servo...");
    denegando = false;
    digitalWrite(PIN_LED_VERDE, LOW);
    digitalWrite(PIN_LED_ROJO,  HIGH);
    servo.write(SERVO_CERRADO);
  }
}


// ════════════════════════════════════════════════════════════════
//  ACCIONES
// ════════════════════════════════════════════════════════════════

void accion_acceso_permitido(String nombre) {

  Serial.println("ACCESO PERMITIDO: " + nombre);

  denegando = false;
  digitalWrite(PIN_LED_ROJO,  LOW);
  digitalWrite(PIN_LED_VERDE, HIGH);

  servo.write(SERVO_ABIERTO);
  puerta_abierta = true;
  t_apertura     = millis();
}


void accion_acceso_denegado() {

  Serial.println("ACCESO DENEGADO");

  if (puerta_abierta) {
    cerrar_puerta();
  }

  digitalWrite(PIN_LED_VERDE, LOW);

  denegando          = true;
  t_denegado         = millis();
  t_parpadeo         = millis();
  led_rojo_encendido = true;
  digitalWrite(PIN_LED_ROJO,  HIGH);
}


void accion_espera() {

  Serial.println("MODO ESPERA");

  denegando = false;

  digitalWrite(PIN_LED_VERDE, LOW);
  digitalWrite(PIN_LED_ROJO,  LOW);

  if (puerta_abierta) {
    cerrar_puerta();
  }
}


// ════════════════════════════════════════════════════════════════
//  AUXILIARES
// ════════════════════════════════════════════════════════════════

void cerrar_puerta() {
  servo.write(SERVO_CERRADO);
  puerta_abierta = false;
  digitalWrite(PIN_LED_VERDE, LOW);
}