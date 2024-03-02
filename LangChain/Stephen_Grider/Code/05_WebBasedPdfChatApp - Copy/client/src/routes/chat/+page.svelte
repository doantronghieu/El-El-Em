<script lang="ts">
	import { beforeNavigate } from '$app/navigation';
	import { resetAll, sendMessage } from '$s/chat/index';
	import ChatPanel from '$c/chat/ChatPanel.svelte';

	beforeNavigate(resetAll);

	function handleSubmit(content: string, useStreaming: boolean) {
		sendMessage({ role: 'user', content }, { useStreaming });
	}
</script>

<div class="chat-panel-wrapper">
	<ChatPanel onSubmit={handleSubmit} />
</div>

<style>
	.chat-panel-wrapper {
		height: 80%;
	}
</style>
