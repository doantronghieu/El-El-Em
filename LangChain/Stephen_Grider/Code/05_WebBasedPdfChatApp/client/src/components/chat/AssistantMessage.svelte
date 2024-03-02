<script lang="ts">
	import { marked } from 'marked';
	import classnames from 'classnames';
	import { scoreConversation } from '$s/chat';
	import Icon from '$c/Icon.svelte';

	export let content = '';
	let score = 0;

	const klass = 'border rounded-full inline-block cursor-pointer hover:bg-slate-200';
	$: upKlass = classnames(klass, {
		'bg-slate-200': score === 1
	});
	$: downKlass = classnames(klass, {
		'bg-slate-200': score === -1
	});

	async function applyScore(_score: number) {
		if (score !== 0) {
			return;
		}
		score = _score;
		return scoreConversation(_score);
	}
</script>

<div class="flex flex-row items-center justify-between">
	<div
		class="message border rounded-md py-1.5 px-2.5 my-0.25 break-words self-start bg-blue-500 text-gray-100"
	>
		{@html marked(content, { breaks: true, gfm: true })}
	</div>
	<div class="flex flex-row flex-1 items-start gap-1 flex-wrap justify-center">
		{#if score >= 0}
			<div class={upKlass} style="line-height: 12px; padding: 6px;">
				<Icon on:click={() => applyScore(1)} name="thumb_up" outlined />
			</div>
		{/if}
		{#if score <= 0}
			<div class={downKlass} style="line-height: 12px; padding: 6px;">
				<Icon on:click={() => applyScore(-1)} name="thumb_down" outlined />
			</div>
		{/if}
	</div>
</div>

<style>
	.message {
		max-width: 80%;
	}
</style>
