#include <avr/wdt.h>
#include <Filters.h> //Easy library to do the calculations
#include <Wire.h>

float testFrequency = 60;                     // test signal frequency (Hz)
float windowLength = 40.0/testFrequency;     // how long to average the signal, for statistist

int Sensor = 0; //Sensor analog input, here it's A0

float intercept = -0.04; // to be adjusted based on calibration testing
float slope = 0.0405; // to be adjusted based on calibration testing
float current_Volts; // Voltage
float voltajeSensor;
float corriente=0;
float Sumatoria=0;
long tiempo=millis();
int N=0;
unsigned long printPeriod = 1000; //Refresh rate
unsigned long previousMillis = 0;
unsigned long previousMillis2 = 0;
const unsigned long intervalo=0;

float get_corriente()
{
  float voltajeSensor;
  float corriente=0;
  float Sumatoria=0;
  long tiempo=millis();
  int N=0;
  while(millis()-tiempo<500)//Duración 0.5 segundos(Aprox. 30 ciclos de 60Hz)
  { 
    voltajeSensor = analogRead(A0)* (1.1 / 1023.0);//voltaje del sensor de corriente alterna
    corriente=voltajeSensor*70.5; //corriente=VoltajeSensor*(100A/1V)
    Sumatoria=Sumatoria+sq(corriente);//Sumatoria de Cuadrados
    N=N+1;
    delay(1);  
  }
  Sumatoria=Sumatoria*2;//Para compensar los cuadrados de los semiciclos negativos.
  corriente=sqrt((Sumatoria)/N); //Ecuación del RMS
  corriente=corriente;
  return(corriente);
}

float get_corriente2()
{
  float voltajeSensor2;
  float corriente2=0;
  float Sumatoria2=0;
  long tiempo2=millis();
  int N2=0;
  while(millis()-tiempo2<500)//Duración 0.5 segundos(Aprox. 30 ciclos de 60Hz)
  { 
    voltajeSensor2 = analogRead(A2)* (1.1 / 1023.0);//voltaje del sensor de corriente alterna
    corriente2=voltajeSensor2*187.0; //corriente=VoltajeSensor*(100A/1V)
    Sumatoria2=Sumatoria2+sq(corriente2);//Sumatoria de Cuadrados
    N2=N2+1;
    delay(1);  
  }
  Sumatoria2=Sumatoria2*2;//Para compensar los cuadrados de los semiciclos negativos.
  corriente2=sqrt((Sumatoria2)/N2); //Ecuación del RMS
  corriente2=corriente2;
  return(corriente2);
}

float get_corriente3()
{
  float voltajeSensor3;
  float corriente3=0;
  float Sumatoria3=0;
  long tiempo3=millis();
  int N3=0;
  while(millis()-tiempo3<500)//Duración 0.5 segundos(Aprox. 30 ciclos de 60Hz)
  { 
    voltajeSensor3 = analogRead(A3)* (1.1 / 1023.0);//voltaje del sensor de corriente alterna
    corriente3=voltajeSensor3*157.0; //corriente=VoltajeSensor*(100A/1V)
    Sumatoria3=Sumatoria3+sq(corriente3);//Sumatoria de Cuadrados
    N3=N3+1;
    delay(1);  
  }
  Sumatoria3=Sumatoria3*2;//Para compensar los cuadrados de los semiciclos negativos.
  corriente3=sqrt((Sumatoria3)/N3); //Ecuación del RMS
  corriente3=corriente3;
  return(corriente3);
}
float get_corriente4()
{
  float voltajeSensor4;
  float corriente4=0;
  float Sumatoria4=0;
  long tiempo4=millis();
  int N4=0;
  while(millis()-tiempo4<500)//Duración 0.5 segundos(Aprox. 30 ciclos de 60Hz)
  { 
    voltajeSensor4 = analogRead(A4)* (1.1 / 1023.0);//voltaje del sensor de corriente alterna
    corriente4=voltajeSensor4*127.0; //corriente=VoltajeSensor*(100A/1V)
    Sumatoria4=Sumatoria4+sq(corriente4);//Sumatoria de Cuadrados
    N4=N4+1;
    delay(1);  
  }
  Sumatoria4=Sumatoria4*2;//Para compensar los cuadrados de los semiciclos negativos.
  corriente4=sqrt((Sumatoria4)/N4); //Ecuación del RMS
  corriente4=corriente4;
  return(corriente4);
}

void setup() {
 Serial.begin(9600); 
 pinMode(7, OUTPUT);      // set the LED pin mode
 wdt_enable(WDTO_8S);
}

void loop() {
unsigned long ahora=millis();
if (ahora - previousMillis2 >= intervalo){ 
 float I1 = get_corriente();
 float I2 = get_corriente2();
 float I3 = get_corriente3();
 float I4 = get_corriente4();
 float Itot = get_corriente2() + get_corriente3();
   
digitalWrite(7, LOW);

RunningStatistics inputStats;                //Easy life lines, actual calculation of the RMS requires a load of coding
inputStats.setWindowSecs( windowLength );
int var=0; 
while( var<3 ) {    
    Sensor = analogRead(A5);  // read the analog in value:
    inputStats.input(Sensor);  // log to Stats function
     
    if((unsigned long)(millis() - previousMillis) >= printPeriod) {
      previousMillis = millis();   // update time every second
      current_Volts = intercept + slope * inputStats.sigma(); //Calibartions for offset and amplitude
      current_Volts= current_Volts*(44.5231);                //Further calibrations for the amplitude
      if (var==0 || var==1){
      var++;
      
      }
      else{    
      var++;}
      
      if (current_Volts<=1){
        current_Volts=0;}
      }
    }
  // Uploads new telemetry to ThingsBoard using MQTT.
  float PCONS = I1*current_Volts;
  digitalWrite(7, HIGH);
  Serial.print(PCONS);
  Serial.print("x");
  Serial.print(current_Volts);
  Serial.print("x");
  Serial.println(I1);

digitalWrite(7, LOW); 
delay(1000);

previousMillis2=ahora;}
wdt_reset();

    }
    
