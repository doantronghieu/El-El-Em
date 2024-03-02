<script lang="ts">
	import type { PageData } from './$types';
	import { beforeNavigate } from '$app/navigation';
	import { resetAll, sendMessage } from '$s/chat/index';
	import PdfViewer from '$c/PdfViewer.svelte';
	import ChatPanel from '$c/chat/ChatPanel.svelte';

	export let data: PageData;

	const document = data.document;
	const documentUrl = data.documentUrl;

	function handleSubmit(content: string, useStreaming: boolean) {
		sendMessage({ role: 'user', content }, { useStreaming, documentId: document.id });
	}

	beforeNavigate(resetAll);
</script>

{#if data.error}
	{data.error}
{/if}

{#if document}
	<div class="grid grid-cols-3 gap-2" style="height: calc(100vh - 80px);">
		<div class="col-span-1">
			<ChatPanel documentId={document.id} onSubmit={handleSubmit} />
		</div>
		<div class="col-span-2">
			<PdfViewer url={documentUrl} />
		</div>
	</div>
{/if}
