# Instructions

To build:
`./docker_build.sh`

To run & connect:
`./docker_connect.sh`

Once connected, to set up MySQL inside of Docker environment:
`mysql ithemal < {{COST_MODEL_FILE.sql}}`

To stop container (run from outside of container):
`./docker_stop.sh` (MySQL data persists)
