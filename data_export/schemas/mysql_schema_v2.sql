-- DROP DATABASE IF EXISTS @db_name;
-- CREATE DATABASE IF NOT EXISTS @db_name;
-- USE @db_name;

-- please refer to the overleaf document for rationale

DROP TABLE IF EXISTS time;
DROP TABLE IF EXISTS time_metadata; 
DROP TABLE IF EXISTS code_metadata;
DROP TABLE IF EXISTS code;
DROP TABLE IF EXISTS config;
DROP TABLE IF EXISTS cpu_desc;


CREATE TABLE cpu_desc (
  arch_id int(11) NOT NULL AUTO_INCREMENT,
  name varchar(255) NOT NULL,
  vendor varchar(255) NOT NULL,
  PRIMARY KEY (arch_id),
  UNIQUE KEY (name, vendor)
);

CREATE TABLE config (
  config_id int(11) NOT NULL AUTO_INCREMENT,
  compiler varchar(255) DEFAULT NULL,
  flags varchar(255) DEFAULT NULL,
  arch_id int(11) DEFAULT NULL,
  PRIMARY KEY (config_id),
  UNIQUE KEY (compiler,flags,arch_id),
  CONSTRAINT arch_id_mapping_config FOREIGN KEY (arch_id) REFERENCES cpu_desc(arch_id)
);

CREATE TABLE code (
  code_id int(32) NOT NULL AUTO_INCREMENT,
  code_raw TEXT, -- raw bytes stored as ascii strings 
  code_intel TEXT,
  code_att TEXT,
  PRIMARY KEY (code_id)
);

-- main metadata table, you can many of your own metadata tables
-- only restriction is that you should have a foreign key to the code table
-- to point to the basic block for which the metadata is for.

CREATE TABLE code_metadata (
  metadata_id int(32) NOT NULL AUTO_INCREMENT,
  config_id int(11) NOT NULL,
  code_id int(32) NOT NULL,
  module varchar(255) NOT NULL,
  rel_addr int(32) NOT NULL,
  function TEXT,
  PRIMARY KEY (metadata_id),
  UNIQUE KEY (code_id),
  CONSTRAINT code_id_mapping_code FOREIGN KEY (code_id) REFERENCES code(code_id),
  CONSTRAINT config_id_mapping_code FOREIGN KEY (config_id) REFERENCES config(config_id)
);


CREATE TABLE time_metadata (
  kind_id int(11) NOT NULL AUTO_INCREMENT,
  name TEXT NOT NULL,
  PRIMARY KEY (kind_id)
);


CREATE TABLE time (
  time_id int(32) NOT NULL AUTO_INCREMENT,

  code_id int(32) NOT NULL,
  arch_id int(11) NOT NULL,
  kind_id int(11) NOT NULL,

  cycle_count int(32) NOT NULL,
  l1drmisses int(11) DEFAULT NULL,
  l1dwmisses int(11) DEFAULT NULL,
  l1imisses int(11) DEFAULT NULL,
  conswitch int(11) DEFAULT NULL,

  PRIMARY KEY (time_id),
  CONSTRAINT code_id_mapping_time FOREIGN KEY (code_id) REFERENCES code(code_id),
  CONSTRAINT arch_id_mapping_time FOREIGN KEY (arch_id) REFERENCES cpu_desc(arch_id),
  CONSTRAINT kind_id_mapping_time FOREIGN KEY (kind_id) REFERENCES time_metadata(kind_id) 
);


