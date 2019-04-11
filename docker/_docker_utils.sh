SCRIPT_DIR="$(dirname $0)"
export HOSTNAME="$(hostname)"

function get_sudo() {
    if ! sudo -S true < /dev/null 2> /dev/null; then
        echo "sudo access required for docker:"
        sudo true
    fi
}

# wrapper around docker-compose to allow this to work from any directory
function docker_compose() {
    sudo docker-compose -f "${SCRIPT_DIR}/docker-compose.yml" "$@"
    return $?
}
