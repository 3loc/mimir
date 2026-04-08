INVENTORY = ansible/inventory.yml
PLAYBOOK  = ansible/deploy-mimir.yml

.PHONY: deploy setup check probe run logs status restart stop

# Idempotent deploy. Assumes passwordless sudo.
deploy:
	ansible-playbook -i $(INVENTORY) $(PLAYBOOK)

# First-time setup — prompts for sudo password.
setup:
	ansible-playbook -i $(INVENTORY) $(PLAYBOOK) --ask-become-pass

# Dry run.
check:
	ansible-playbook -i $(INVENTORY) $(PLAYBOOK) --check --diff

# Smoke-test the running daemon.
probe:
	@echo "=== mimir.service ==="
	@systemctl is-active mimir.service || true
	@systemctl --no-pager --lines=15 status mimir.service 2>&1 | head -25 || true
	@echo
	@echo "=== TCP probe (3 seconds of transcript on tcp://127.0.0.1:7200) ==="
	@python3 -c "import socket, time; \
s = socket.socket(); s.settimeout(4); s.connect(('127.0.0.1', 7200)); \
end = time.time() + 3; \
buf = b''; \
[buf := buf + (s.recv(4096) or b'') for _ in iter(int, 1) if time.time() < end]; \
print(buf.decode('utf-8', errors='replace') or '(no lines emitted in 3s)')" \
		2>&1 || echo "(probe failed — is mimir running?)"

# Run the daemon in the foreground for interactive testing.
run:
	.venv/bin/python src/mimir.py

# Tail the journal.
logs:
	journalctl -u mimir.service -f --no-pager

status:
	systemctl status mimir.service --no-pager

restart:
	sudo systemctl restart mimir.service

stop:
	sudo systemctl stop mimir.service
