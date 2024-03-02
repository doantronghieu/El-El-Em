<script lang="ts">
	import { onMount } from 'svelte';
	import {
		store,
		resetError,
		fetchConversations,
		createConversation,
		getActiveConversation
	} from '$s/chat';
	import Alert from '$c/Alert.svelte';
	import ChatInput from '$c/chat/ChatInput.svelte';
	import ChatList from '$c/chat/ChatList.svelte';
	import ConversationSelect from '$c/chat/ConversationSelect.svelte';

	export let onSubmit: (text: string, useStreaming: boolean) => void;
	export let documentId: number;

	let useStreaming = !!localStorage.getItem('streaming');

	$: localStorage.setItem('streaming', useStreaming ? 'true' : '');
	$: activeConversation = $store.activeConversationId ? getActiveConversation() : null;

	function handleSubmit(event: CustomEvent<string>) {
		if (onSubmit) {
			onSubmit(event.detail, useStreaming);
		}
	}

	function handleNewChat() {
		createConversation(documentId);
	}

	onMount(() => {
		fetchConversations(documentId);
	});
</script>

<div
	style="height: calc(100vh - 80px);"
	class="flex flex-col h-full bg-slate-50 border rounded-xl shadow"
>
	<div class="rounded-lg border-b px-3 py-2 flex flex-row items-center justify-between">
		<div class="opacity-40">
			<input id="chat-type" type="checkbox" bind:checked={useStreaming} />
			<label for="chat-type" class="italic">Streaming</label>
		</div>
		<div class="flex gap-2">
			<ConversationSelect conversations={$store.conversations} />
			<button class="rounded text-sm border border-blue-500 px-2 py-0.5" on:click={handleNewChat}
				>New Chat</button
			>
		</div>
	</div>
	<div class="flex flex-col flex-1 px-3 py-2 overflow-y-scroll">
		<ChatList messages={activeConversation?.messages || []} />
		<div class="relative">
			{#if $store.error && $store.error.length < 200}
				<div class="p-4">
					<Alert type="error" onDismiss={resetError}>
						{$store.error}
					</Alert>
				</div>
			{/if}
			<ChatInput on:submit={handleSubmit} />
		</div>
	</div>
</div>

<style>
</style>
