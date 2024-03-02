<script lang="ts">
	export let type: string = 'text';
	export let value = '';

	function typeAction(node: HTMLInputElement) {
		node.type = type;
	}
</script>

<input
	use:typeAction
	bind:value
	class="py-3 px-4 block w-full border border-gray-200 rounded-md text-sm focus:border-blue-500 focus:ring-blue-500"
	required
/>
