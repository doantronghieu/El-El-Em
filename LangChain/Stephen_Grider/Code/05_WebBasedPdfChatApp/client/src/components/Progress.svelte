<script>
	import { tweened } from 'svelte/motion';

	export let progress = 0;

	const width = tweened(1, {
		duration: 1000
	});

	$: width.set(progress);
</script>

{#if $$slots.default && $width === 100}
	<slot />
{:else}
	<div class="flex my-3 w-full h-1.5 bg-gray-200 rounded-full overflow-hidden dark:bg-gray-700">
		<div
			class="flex flex-col justify-center overflow-hidden bg-blue-800"
			role="progressbar"
			style="width: {$width}%"
		/>
	</div>
{/if}
