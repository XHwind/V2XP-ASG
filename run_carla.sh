#!/usr/bin/env bash
World_port=2000
Help() {
  echo "Running the CARLA with run/kill/re-run loop"
  echo "Syntax: scriptTemplate [-h|p]"
  echo "Options: "
  echo "h    Display the help and exit"
  echo "p    CARLA port number. Default=$World_port"
}
# Get the options
while getopts ":hp:" option; do
  case $option in
  h) # display Help
    Help
    exit
    ;;
  p) # Enter a name
    World_port=$OPTARG ;;
  \?) # Invalid option
    echo "Error: Invalid option"
    exit
    ;;
  esac
done

count=0
while true; do
  echo "CarlaUE4.sh --world-port=$World_port"
  bash -c "exec -a CarlaUE4_$World_port timeout 1000 CARLA_0.9.12/CarlaUE4.sh --world-port=$World_port"
  sleep 5
  pkill -f CarlaUE4_$World_port
  ((count += 1))
  res=$(($count % 5))
  if [[ $res -eq 0 ]]; then
    pkill CarlaUE4
  fi
  echo "CARLA failed wait for 10 seconds before running"
  sleep 10

done
