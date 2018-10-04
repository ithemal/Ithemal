SCRIPT_DIR="$(dirname $0)"
# wrapper around docker-compose to allow this to work from any directory
function docker_compose() {
    sudo docker-compose -f "${SCRIPT_DIR}/docker-compose.yml" "$@"
    return $?
}
