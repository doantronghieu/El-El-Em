<script lang="ts">
	import AssistantMessage from '$c/chat/AssistantMessage.svelte';
	import UserMessage from '$c/chat/UserMessage.svelte';
	import PendingMessage from '$c/chat/PendingMessage.svelte';

	interface Message {
		role: 'user' | 'system' | 'assistant' | 'pending' | 'human' | 'ai';
		content: string;
	}
	export let messages: Message[] = [];

	const scrollIntoView = (node: HTMLDivElement, _m: any) => {
		setTimeout(() => {
			node.scrollIntoView();
		}, 0);
		return {
			update: () => node.scrollIntoView()
		};
	};
</script>

<div class="overflow-y-auto flex flex-col flex-1">
	<div class="flex flex-col flex-1 gap-3 px-1.5 py-1">
		{#each messages as message}
			{#if message.role === 'user' || message.role === 'human'}
				<UserMessage content={message.content} />
			{:else if message.role === 'assistant' || message.role === 'ai'}
				<AssistantMessage content={message.content} />
			{:else if message.role === 'pending'}
				<PendingMessage />
			{/if}
		{/each}
	</div>
	<div class="pt-4" use:scrollIntoView={messages} />
</div>
