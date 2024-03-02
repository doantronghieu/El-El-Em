<script lang="ts">
	import type { PageData } from './$types';
	import BarChart from '$c/BarChart.svelte';
	import AuthGuard from '$c/AuthGuard.svelte';

	export let data: PageData;

	$: llmScores = data.scores && data.scores['llm'];
	$: retrieverScores = data.scores && data.scores['retriever'];
	$: memoryScores = data.scores && data.scores['memory'];
</script>

<AuthGuard />
<div class="w-1/2">
	<h2 class="text-3xl font-bold m-3">LLM Scores</h2>
	{#if llmScores}
		<BarChart startingColor={{ r: 0, g: 99, b: 132 }} data={llmScores} />
	{/if}
</div>

<hr class="m-10" />

<div class="w-1/2">
	<h2 class="text-3xl font-bold m-3">Retriever Scores</h2>
	{#if retrieverScores}
		<BarChart startingColor={{ r: 255, g: 99, b: 132 }} data={retrieverScores} />
	{/if}
</div>

<hr class="m-10" />

<div class="w-1/2">
	<h2 class="text-3xl font-bold m-3">Memory Scores</h2>
	{#if memoryScores}
		<BarChart startingColor={{ r: 30, g: 200, b: 60 }} data={memoryScores} />
	{/if}
</div>
