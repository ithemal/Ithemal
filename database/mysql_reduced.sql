DROP DATABASE IF EXISTS training;
CREATE DATABASE IF NOT EXISTS training;
USE training;

DROP TABLE IF EXISTS config;
CREATE TABLE config (
  config_id int(11) NOT NULL AUTO_INCREMENT,
  compiler varchar(255) DEFAULT NULL,
  flags varchar(255) DEFAULT NULL,
  PRIMARY KEY (config_id),
  UNIQUE KEY (compiler,flags)
);


DROP TABLE IF EXISTS code;
CREATE TABLE code (
  code_id int(11) NOT NULL AUTO_INCREMENT,
  program varchar(255) NOT NULL,
  config_id int(11) NOT NULL,
  rel_addr int(32) NOT NULL,
  code TEXT,
  PRIMARY KEY (code_id),
  UNIQUE KEY (program,config_id,rel_addr),
  CONSTRAINT code_ibfk_1 FOREIGN KEY (config_id) REFERENCES config(config_id)
);



DROP TABLE IF EXISTS times;
CREATE TABLE times (
  time_id int(11) NOT NULL AUTO_INCREMENT,
  code_id int(11) NOT NULL,
  arch int(11) NOT NULL,
  time int(32) NOT NULL,
  PRIMARY KEY (time_id),
  CONSTRAINT times_ibfk_1 FOREIGN KEY (code_id) REFERENCES code(code_id)
);
