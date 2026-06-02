#pragma once

#include "Protocol.h"

class IPCClient
{
public:
	~IPCClient();

	void Connect();
	bool TryConnect(int maxRetries = 5, int initialDelayMs = 200);
	bool IsConnected() const;
	void Disconnect();
	protocol::Response SendBlocking(const protocol::Request &request);

	void Send(const protocol::Request &request);
	protocol::Response Receive();

private:
	HANDLE pipe = INVALID_HANDLE_VALUE;
};