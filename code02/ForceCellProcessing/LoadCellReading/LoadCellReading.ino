// Global Variables
  // Load Cell 1
unsigned long volt1; // array
 // Load Cell 2
unsigned long volt2; // array
 // Sent Data
unsigned long Send = 0;
int sync = 0;
int count = 0;
int syncread = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(A8, INPUT);
  pinMode(A9, INPUT);
  //pinMode(65, OUTPUT);
  pinMode(A11, INPUT);
}

void loop() {
//  if (count < 1000){
//    count += 1;
//  }
//  else {
//    sync = 1;
//    digitalWrite(65, sync);
//  }


  sync = digitalRead(A11);



  
  volt1 = analogRead(A8);
  volt2 = analogRead(A9);
  // Send via serial to python
  Send = (volt1*10000)+ (volt2 * 10) + sync;
  Serial.println(Send);


  delay(10);
}
